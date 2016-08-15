
/***
 * Standard reverse-mode automatic differentiation system.
 * Uses backwards diff to compute derivative WRT some variable at a given interval.
 */

use std::f32;
use std::vec;
use std::collections::HashMap;
use std::collections::hash_set::HashSet;
use rand::Rng;

extern crate ocl;
use ocl::{ProQue, Buffer};
extern crate rand;

static KERNEL_SOURCE: &'static str = include_str!("kernel_source.cl");

type Dimension = (usize, usize);
type NodeId = usize;

enum Operation {
	Input,
	MatrixMultiply(NodeId, NodeId),
	ConstantAdd(NodeId, f32),
	ConstantMultiply(NodeId, f32),
	ElementAdd(NodeId, NodeId),
	ElementMultiply(NodeId, NodeId),
	ElementInverse(NodeId),

	Unary(NodeId, Box<Fn(f32)->f32>, Box<Fn(f32)->f32>), // x, f, df/dx
	Binary(NodeId, NodeId, Box<Fn(f32,f32)->f32>, Box<Fn(f32,f32)->f32>, Box<Fn(f32,f32)->f32>), // LHS, RHS, F, dF/dLHS, dF/dRHS
}

struct Node {
	id : NodeId,
	shape : Dimension,
	operation : Operation,
}

struct Graph {
//	proque : ProQue,
	nodes : Vec<Node>,
}

impl Graph {
	fn new() -> Graph {
		Graph {
//			proque : ProQue::builder().src(KERNEL_SOURCE).dims([3]).build().unwrap(),
			nodes : vec![],
		}
	}

	// Graph methods
	fn get_output(&self, node_id : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		match self.nodes[node_id].operation {
			Operation::Input => { input_map.get(&node_id).unwrap().clone() },
			Operation::MatrixMultiply(n1, n2) => {
				// Verify shapes
				let a_width = self.nodes[n1].shape.1; // Columns (j)
				let a_height = self.nodes[n1].shape.0; // Rows (i)
				let b_width = self.nodes[n2].shape.1; // Columns (k)
				let b_height = self.nodes[n2].shape.0; // Rows (j)
				let c_width = b_width;
				let c_height = a_height;
				assert_eq!(a_width, b_height);

				let mut result = vec![0.0; a_height*b_width];

				// Set up the buffer for this node.
				let a : Vec<f32> = self.get_output(n1, &input_map);
				let b : Vec<f32> = self.get_output(n2, &input_map);

				// Multiply
				for i in 0..a_height { // Column [Iterating over row]
					for k in 0..b_width { // Row/Width [Iterating over column]
						let mut accumulator = 0.0;
						for j in 0..a_width { // Column
							accumulator += a[j + i*a_width]*b[k + j*b_width];
						}
						result[k + i*c_width] = accumulator;
					}
				}

				result
			},
			Operation::ConstantAdd(node, scalar) => {
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]+scalar;
				}
				result
			},
			Operation::ConstantMultiply(node, scalar) => {
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]*scalar;
				}
				result
			},
			Operation::ElementAdd(node_a_id, node_b_id) => { 
				let a : Vec<f32> = self.get_output(node_a_id, &input_map);
				let b : Vec<f32> = self.get_output(node_b_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]+b[i];
				}
				result
			},
			Operation::ElementMultiply(node_a_id, node_b_id) => {
				let a : Vec<f32> = self.get_output(node_a_id, &input_map);
				let b : Vec<f32> = self.get_output(node_b_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]*b[i];
				}
				result
			},
			Operation::ElementInverse(node) => {
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = 1.0/a[i];
				}
				result
			},
			Operation::Unary(node_id, ref f, _) => {
				let a : Vec<f32> = self.get_output(node_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = f(a[i]);
				}
				result
			},
			Operation::Binary(node_a_id, node_b_id, ref f, _, _) => { // LHS, RHS, F, dF/dLHS, dF/dRHS
				let a : Vec<f32> = self.get_output(node_a_id, &input_map);
				let b : Vec<f32> = self.get_output(node_b_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = f(a[i], b[i]);
				}
				result
			},
		}
	}

	fn get_output_with_gradient(&self, node_id : NodeId, wrt : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
		// If we use infintesimals when calculating forward gradient, we get these:
		// (x + x'eps) + (y + y'eps) = (x+y, (x'+y')eps)
		// (x + x'eps) * (y + y'eps) = xy + xy'eps + x'yeps + x'y'eps^2 = xy + (xy' + x'y)eps
		// If this is the derivative of something WRT (this node), the epsilon residual is 1.0.
		// ELSE it's 0.
		// In general, g(<u,u'>, <v, v'>) = <g(u,v) , dg/du(u,v)*u' + dg/dv(u,v)*v'>

		match self.nodes[node_id].operation {
			Operation::Input => {
				let len = self.nodes[node_id].shape.0 * self.nodes[node_id].shape.1;
				if node_id == wrt {
					(input_map.get(&node_id).unwrap().clone(), vec![1.0; len])
				} else {
					(input_map.get(&node_id).unwrap().clone(), vec![0.0; len])
				}
			},
			Operation::MatrixMultiply(n1, n2) => {
				// Verify shapes
				let a_width = self.nodes[n1].shape.1; // Columns (j)
				let a_height = self.nodes[n1].shape.0; // Rows (i)
				let b_width = self.nodes[n2].shape.1; // Columns (k)
				let b_height = self.nodes[n2].shape.0; // Rows (j)
				let c_width = b_width;
				let c_height = a_height;
				assert_eq!(a_width, b_height);

				let (a_real, a_residual) = self.get_output_with_gradient(n1, wrt, &input_map);
				let (b_real, b_residual) = self.get_output_with_gradient(n2, wrt, &input_map);

				// d(XY) = (dX)Y + X(dY)

				let mut result = vec![0.0; a_real.len()];
				let mut residual = vec![0.0; a_real.len()];

				// Multiply n1 * n2 and dn1*n2 + n1*dn2 at the same time.
				for i in 0..a_height { // Column [Iterating over row]
					for k in 0..b_width { // Row/Width [Iterating over column]
						let mut accumulator = 0.0;
						let mut residual_accumulator = 0.0;
						for j in 0..a_width { // Column
							let a_pos = j+i*a_width;
							let b_pos = k+j*b_width;
							accumulator += a_real[a_pos]*b_real[b_pos];
							residual_accumulator += a_real[a_pos]*b_residual[b_pos] + a_residual[a_pos]*b_real[b_pos];
						}
						result[k + i*c_width] = accumulator;
						// Is the derivative of a matrix with respect to itself the identity or ones?
						residual[k + i*c_width] = residual_accumulator;
					}
				}

				(result, residual)
			},
			Operation::ConstantAdd(node, scalar) => {
				let (mut real, mut residual) = self.get_output_with_gradient(node, wrt, &input_map);

				let vec_len = self.nodes[node_id].shape.0*self.nodes[node_id].shape.1;
				
				// First, compute the epsilon values.
				// (x + x'eps) + (scalar, 0eps) with broadcast.
				for i in 0..vec_len {
					real[i] += scalar;
					//residual[i] += 0; // Since this is a const.
				}

				(real, residual)
			},
			Operation::ConstantMultiply(node, scalar) => {
				// (x + x'eps) * (y + y'eps) -> (xy) + (xy'eps) + (x'yeps) + (x'y'eps^2) -> (xy) + (xy'eps) + (x'yeps)
				// (xy) + (xy' + x'y)eps
				// Since y is a constant scalar, yeps = 0 and y' is zero.
				// (x + x_res) * (y + 0) -> (xy + x_res*y)
				let (mut real, mut residual) = self.get_output_with_gradient(node, wrt, &input_map);
				for i in 0..real.len() {
					real[i] *= scalar;
					residual[i] *= scalar;
				}
				(real, residual)
			},
			Operation::ElementAdd(node_a_id, node_b_id) => { 
				// (x + x'eps) + (y + y'eps) = (x+y, (x'+y')eps)
				let (a_real, a_residual) = self.get_output_with_gradient(node_a_id, wrt, &input_map);
				let (b_real, b_residual) = self.get_output_with_gradient(node_b_id, wrt, &input_map);
				let mut real = vec![0.0; a_real.len()];
				let mut residual = vec![0.0; a_real.len()];
				for i in 0..a_real.len() {
					real[i] = a_real[i]+b_real[i];
					residual[i] = a_residual[i]+b_residual[i];
				}
				(real, residual)
			},
			Operation::ElementMultiply(node_a_id, node_b_id) => {
				// (x + x'eps) * (y + y'eps) = xy + xy'eps + x'yeps + x'y'eps^2 = xy + (xy' + x'y)eps
				let (a_real, a_residual) = self.get_output_with_gradient(node_a_id, wrt, &input_map);
				let (b_real, b_residual) = self.get_output_with_gradient(node_b_id, wrt, &input_map);
				let mut real = vec![0.0; a_real.len()];
				let mut residual = vec![0.0; a_real.len()];
				for i in 0..a_real.len() {
					real[i] = a_real[i]*b_real[i];
					residual[i] = a_real[i]*b_residual[i] + a_residual[i]*b_real[i];
				}
				(real, residual)
			},
			Operation::ElementInverse(node) => {
				// <u, u'> / <v, v'> = < u/v, u'v - uv'/v^2>
				// Let u = (1.0, 0.0)
				// u' = 0.0
				// v = a_real
				// v' = a_res
				// u'v - uv' -> -1.0*a_res / a_real*a_real
				let (a_real, a_res) = self.get_output_with_gradient(node, wrt, &input_map);
				let mut result = vec![0.0; a_real.len()];
				let mut residual = vec![0.0; a_real.len()];
				for i in 0..a_real.len() {
					result[i] = 1.0/a_real[i];
					residual[i] = (-1.0*a_res[i])/(a_real[i]*a_real[i]);
				}
				(result, residual)
			},
			Operation::Unary(node_id, ref f, ref dfdx) => {
				let (a_real, a_res) = self.get_output_with_gradient(node_id, wrt, &input_map);
				let mut result = vec![0.0; a_real.len()];
				let mut residual = vec![0.0; a_real.len()];
				for i in 0..a_real.len() {
					result[i] = f(a_real[i]);
					residual[i] = dfdx(a_real[i])*a_res[i];
				}
				(result, residual)
			},
			Operation::Binary(node_a_id, node_b_id, ref f, ref dfda, ref dfdb) => { // LHS, RHS, F, dF/dLHS, dF/dRHS
				// g( <a_real, a_res>, <b_real, b_res> ) = < g(a_real, b_real), dgda(a_real, b_real)*da + dgdb(a_real, b_real)*db>
				let (a_real, a_res) = self.get_output_with_gradient(node_a_id, wrt, &input_map);
				let (b_real, b_res) = self.get_output_with_gradient(node_b_id, wrt, &input_map);
				let mut result_real = vec![0.0; a_real.len()];
				let mut result_residual = vec![0.0; a_real.len()];
				for i in 0..a_real.len() {
					result_real[i] = f(a_real[i], b_real[i]);
					result_residual[i] = dfda(a_real[i], b_real[i])*a_res[i] + dfdb(a_real[i], b_real[i])*b_res[i];
				}
				(result_real, result_residual)
			},
		}
	}

	// Node creation
	fn input(&mut self, shape : Dimension) -> NodeId {
		let mut n = Node {
			id : 0,
			shape : shape,
			operation : Operation::Input,
		};
		self.nodes.push(n);
		let id = self.nodes.len()-1;
		n.id = id;
		id
	}

	fn insert_op(&mut self, shape_ref : NodeId, op : Operation) -> NodeId {
		let mut n = Node {
			id : self.nodes.len(),
			shape : self.nodes[shape_ref].shape,
			operation : op
		};
		let id = n.id;
		self.nodes.push(n);
		id
	}

	fn insert_op_with_shape(&mut self, shape : Dimension, op : Operation) -> NodeId {
		let mut n = Node {
			id : self.nodes.len(),
			shape : shape,
			operation : op
		};
		let id = n.id;
		self.nodes.push(n);
		id
	}

	fn matmul(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId {
		let mut n = Node {
			id : 0,
			shape : (self.nodes[node_a_id].shape.0, self.nodes[node_b_id].shape.1),
			//operation : Operation::BinaryElement(node_a_id, node_b_id, Box::new(|a, b| { 0.0 })),
			operation : Operation::MatrixMultiply(node_a_id, node_b_id),
		};
		self.nodes.push(n);
		let id = self.nodes.len()-1;
		n.id = id;
		id
	}

	// Convenience methods:
	fn inverse(&mut self, node_id : NodeId) -> NodeId {
		self.insert_op(node_id, Operation::ElementInverse(node_id))
	}

	fn constant_add(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::ConstantAdd(node_id, scalar))
	}

	fn constant_multiply(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::ConstantMultiply(node_id, scalar))
	}

	fn power(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::Unary(
			node_id,
			Box::new(move |x| { x.powf(scalar) }),
			Box::new(move |x| { scalar*x.powf(scalar-1.0) }),
		))
	}

	fn add(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId {
		self.insert_op(node_a_id, Operation::ElementAdd(node_a_id, node_b_id))
	}

	fn hadamard_product(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId { // Schur/Element-wise product.
		assert_eq!(self.nodes[node_a_id].shape, self.nodes[node_b_id].shape);
		self.insert_op(node_a_id, Operation::ElementMultiply(node_a_id, node_b_id))
	}

	fn sigmoid(&mut self, node_id : NodeId) -> NodeId {
		self.insert_op(node_id, Operation::Unary(
			node_id, 
			Box::new(|x| { 1.0/(1.0 + (-x).exp()) }), 
			Box::new(|x| { 1.0/(1.0 + (-x).exp()) * (1.0 - (1.0/(1.0 + (-x).exp()))) }) // if f(x) = sigmoid, df/dx = f(x) * (1-f(x))
		))
	}
}

#[cfg(test)]
mod tests {
	extern crate rand;
	use super::{Graph, Dimension, Node, NodeId};
	use std::collections::HashMap;
	use rand::Rng;

	#[test]
	fn test_identity() {
		// Build graph
		let mut input = HashMap::new();
		let mut g = Graph::new();
		let m : NodeId = g.input((3, 3));
		let n : NodeId = g.input((1, 3));
		let o = g.matmul(n, m);
		let j : NodeId = g.input((3, 5));
		let k = g.matmul(m, j);

		// Shape check
		assert_eq!(g.nodes[m].shape, (3, 3));
		assert_eq!(g.nodes[n].shape, (1, 3));
		assert_eq!(g.nodes[o].shape, (1, 3));

		// Configure data
		input.insert(m, vec![
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0
		]);
		input.insert(n, vec![
			0.0, 1.0, 2.0
		]);
		input.insert(j, vec![
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,
			10.0, 11.0, 12.0,
			13.0, 14.0, 15.0
		]);

		// Verify outputs
		let res = g.get_output(o, &input);
		assert_eq!(res[0], 0.0);
		assert_eq!(res[1], 1.0);
		assert_eq!(res[2], 2.0);
		let res2 = g.get_output(k, &input);
		for i in 0..15 {
			assert_eq!(res2[i], (i as f32+1.0));
		}
	}

	#[test]
	fn test_autodiff_simple() {
		let mut g = Graph::new();
		let a = g.input((2, 3));
		let b = g.constant_add(a, 10.0);
		let c = g.constant_multiply(a, 2.0);
		
		let mut input = HashMap::new();
		input.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let res = g.get_output(b, &input);
		let (a_val, db_val_wrt_a) = g.get_output_with_gradient(b, a, &input);
		let (a_val_2, db_val_wrt_b) = g.get_output_with_gradient(b, b, &input);
		let (a_val_3, dc_val_wrt_a) = g.get_output_with_gradient(c, a, &input);
		assert_eq!(a_val[0], 11.0);
		assert_eq!(a_val_2[0], 11.0);
		assert_eq!(db_val_wrt_a[0], 1.0); // d/dx x+10 = 1
		assert_eq!(db_val_wrt_b[0], 0.0); // d/dy x+10 = 0
		assert_eq!(dc_val_wrt_a[0], 2.0); // d/dx x*2 = 2  
	}

	#[test]
	fn test_autodiff_complex() {
		const EPSILON : f32= 0.0001;
		let mut g = Graph::new();
		// f(x) = x^3 + 3*x - 1.0/x + yx + y + 10
		// TODO: 2^x and e^x and y^x
		// df/dx (x) = 3*x^2 + 3 + 1/x^2 + y
		let x = g.input((1, 10)); // One row, ten column input.
		let y = g.input((1, 10));
		// x^3 node 1.
		let a = g.power(x, 3.0);

		// 3*x
		let b = g.constant_multiply(x, 3.0);

		// 1.0/x
		let c = g.inverse(x);

		// y*x
		let d = g.hadamard_product(x, y);

		// (x^3 + 3*x)
		let ab = g.add(a, b);

		// (-1.0/x + yx)
		let c2 = g.constant_multiply(c, -1.0);
		let cd = g.add(c2, d);

		// (x^3 + 3*x + -1.0/x + yx)
		let abcd = g.add(ab, cd);

		// ... + y
		let abcde = g.add(abcd, y);
		let abcdef = g.constant_add(abcde, 10.0);
		
		let mut input = HashMap::new();
		input.insert(x, vec![-10.0, -1.0, -0.1, 0.000001, 0.1, 1.0, 10.0, 8.0, 9.0, 1.0]);
		input.insert(y, vec![-5.0, -2.0, -0.5, 1.0, 0.5, 2.0, 5.0, 8.0, 9.0, 1.0]);

		let expected_result = vec![-974.9000000000001, 7.0, 19.249000000000002, -999988.999996, 0.8510000000000009, 17.0, 1094.9, 617.875, 855.8888888888889, 15.0];
		let expected_grad = vec![298.01, 5.0, 102.52999999999999, 1000000000004.0, 103.52999999999999, 9.0, 308.01, 203.015625, 255.01234567901236, 8.0];

		let (fx, dfdx) = g.get_output_with_gradient(abcdef, x, &input);
		for i in 0..10 {
			assert!((fx[i] - expected_result[i]).abs() <= EPSILON);
			assert!((dfdx[i] - expected_grad[i]).abs() <= EPSILON);
		}
	}

	#[test]
	fn numerical_gradient_check() {
		const DELTA : f32 = 0.001;
		const EPSILON : f32 = 0.01;
		let mut g = Graph::new();
		let x = g.input((1, 3));

		// This can be whatever method.
		let op = g.sigmoid(x);

		let mut input = HashMap::new();
		for i in -100..100 {
			let j = (i as f32)/50.0;
			let x0 = j-DELTA;
			let x1 = j;
			let x2 = j+DELTA;

			input.insert(x, vec![x0, x1, x2]);

			let (out, outgrad) = g.get_output_with_gradient(op, x, &input);
			let numerical_grad = (out[2] - out[0]) / (x2 - x0);

			//println!("{:?} : {:?} {:?}", &[x0, x1, x2], &out, &outgrad);
			println!("f({}) -- Calculated gradient : {}.  Numerical approximation : {}.", j, outgrad[1], numerical_grad);
			assert!( (outgrad[1]-numerical_grad).abs() < EPSILON );
		}
	}

	#[test]
	fn test_matrix_multiply_gradient() {
		let mut g = Graph::new();
		let x = g.input((4, 4));
		let y = g.input((4, 4));

		let mut input : HashMap<usize, Vec<f32>> = HashMap::new();
		let mm = g.matmul(x, y);
		let x_data: Vec<f32> = (0..16u32).map(|i| i as f32).collect(); // Get rid of the map for just an array of ints.
		let y_data: Vec<f32> = (0..16u32).map(|i| { 8.0 - i as f32 } ).collect(); 
		
		input.insert(x, x_data.clone());
		input.insert(y, y_data.clone());
		let d_wrt_x = g.get_output_with_gradient(mm, x, &input);
		let d_wrt_y = g.get_output_with_gradient(mm, y, &input);

		/*
		println!("X:{:?}", x_data);
		println!("Y:{:?}", y_data);
		println!("d x*y wrt X:{:?}", d_wrt_x.1);
		println!("d x*y wrt Y:{:?}", d_wrt_y.1);
		let diff : Vec<f32>= (0..16 as usize).map(|i| (d_wrt_x.1[i] - y_data[i]).abs()).collect();
		println!("diff d x*y wrt x vs y: {:?}", diff);
		assert!(d_wrt_x.1 == y_data);
		assert!(d_wrt_y.1 == x_data);
		*/
	}

	#[test]
	fn test_backprop() {
		// Dataset generator.
		let mut rng = rand::thread_rng();

		// Define graph
		let mut g = Graph::new();

		// Define inputs and variables.
		let x = g.input((1, 2));
		let y = g.input((1, 1)); // Our truth for training.
		let w_ih = g.input((2, 5));
		let w_ho = g.input((5, 1));

		let mut w_ih_data : Vec<f32> = (0u32..(2*5)).map(|i| i as f32 / 1000.0).collect();
		let mut w_ho_data : Vec<f32> = (0u32..(5*1)).map(|i| i as f32 / 100.0).collect();

		// Define operations.
		let hidden_z = g.matmul(x, w_ih);
		let hidden_a = g.sigmoid(hidden_z);
		let out = g.matmul(hidden_a, w_ho);

		// Define squared error.
		let inverse_result = g.constant_multiply(y, -1.0);
		let error_product = g.add(inverse_result, out);
		let cost = g.power(error_product, 2.0);

		let mut inputs = HashMap::new();
		for _ in 0..1 {
			// Train an example.
			let x0 : bool = rng.gen();
			let x1 : bool = rng.gen();
			inputs.insert(x, vec![if x0 { 1.0 } else { 0.0 }, if x1 { 1.0 } else { 0.0 }]);
			inputs.insert(y, vec![if (x0 ^ x1) { 1.0 } else { 0.0 }]);
			inputs.insert(w_ih, w_ih_data.clone());
			inputs.insert(w_ho, w_ho_data.clone());
			let (output, grad_wrt_woh) = g.get_output_with_gradient(cost, w_ho, &inputs);
			// TODO: If we set both input infinitesimals to 1, can we do multiple gradients in one pass?
			println!("Error: {:?}.  Delta weights: {:?}", output, grad_wrt_woh);
		}
	}
}
