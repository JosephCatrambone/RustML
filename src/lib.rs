
/***
 * Standard reverse-mode automatic differentiation system.
 * Uses backwards diff to compute derivative WRT some variable at a given interval.
 */

use std::f32;
use std::vec;
use std::collections::HashMap;
use std::collections::hash_set::HashSet;
use rand::Rng;

//extern crate ocl;
//use ocl::{ProQue, Buffer};
extern crate rand;

static KERNEL_SOURCE: &'static str = include_str!("kernel_source.cl");

type Dimension = (usize, usize); // Row/Col.  Height/Width.
type NodeId = usize;

enum Operation {
	Input,
	MatrixMultiply(NodeId, NodeId),
	Unary(NodeId, Box<Fn(f32)->f32>, Box<Fn(f32)->f32>), // x, f, df/dx
	Binary(NodeId, NodeId, Box<Fn(f32,f32)->f32>, Box<Fn(f32,f32)->f32>, Box<Fn(f32,f32)->f32>), // LHS, RHS, F, dF/dLHS, dF/dRHS
}

struct Node {
	id : NodeId,
	shape : Dimension,
	operation : Operation,
}

fn apply_gradient(weight_matrix : &mut Vec<f32>, activation : &Vec<f32>, gradient : &Vec<f32>, error : &Vec<f32>, learning_rate : f32) {
	// Activation of the input layer
	let width = error.len(); // Num columns.
	let height = gradient.len()/width;
	// Delta E / delta w_ji = learning rate * (true_j - output_j) * g'(h_j) * x_i
	for y in 0..height {
		for x in 0..width {
			println!("DEBUG: wm[{}] += {} * {} * {} * {}", x+y*width, learning_rate, gradient[x+y*width] , error[x] , activation[y]);
			weight_matrix[x + y*width] += learning_rate * gradient[x + y*width] * error[x] * activation[y]; // TODO: Multiply by activation at the layer below.;
			println!("DEBUG: wm[{}] = {}", x+y*width, weight_matrix[x+y*width]);
		}
	}
}

fn get_element_from_tensor(matrix : Vec<f32>, shape : Dimension, x : usize, y : usize) -> f32 { // x = column = j.  y = row = i
	matrix[x + y*shape.1]
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
	fn get_shape(&self, node_id : NodeId) -> Dimension {
		self.nodes[node_id].shape.clone()
	}

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

				//let mut result = vec![0.0; a_height*b_width];
				let mut result = vec![0.0; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1];

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
				assert_eq!(a.len(), b.len());
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = f(a[i], b[i]);
				}
				result
			},
		}
	}

	fn get_derivative(&self, node_id : NodeId, wrt : &[NodeId], input_map : &HashMap<NodeId, Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
		// TODO: Memoize this to speed it up.

		// If we use infintesimals when calculating forward gradient, we get these:
		// (x + x'eps) + (y + y'eps) = (x+y, (x'+y')eps)
		// (x + x'eps) * (y + y'eps) = xy + xy'eps + x'yeps + x'y'eps^2 = xy + (xy' + x'y)eps
		// If this is the derivative of something WRT (this node), the epsilon residual is 1.0.
		// ELSE it's 0.
		// In general, g(<u,u'>, <v, v'>) = <g(u,v) , dg/du(u,v)*u' + dg/dv(u,v)*v'>

		//println!("DEBUG: get_derivative({:?}, {:?}, {:?}, {:?}, {:?})", node_id, wrt, input_map, activations, gradients);
		let mut wrt_this : bool = false; 
		for i in 0..wrt.len() { if wrt[i] == node_id { wrt_this = true; } }

		match self.nodes[node_id].operation {
			Operation::Input => {
				let len = self.nodes[node_id].shape.0 * self.nodes[node_id].shape.1;

				// Yes, in theory it's slower to iterate over an array than it is to use a hashmap, but because we know WRT is going to be small, it performs BETTER.

				(
					input_map.get(&node_id).unwrap().clone(),
					if wrt_this {
						vec![1.0; len]
					} else {
						vec![0.0; len]
					}
				)
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

				// d(XY) = (dX)Y + X(dY)
				let mut result = vec![0.0; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1];
				let mut residual = vec![0.0; result.len()];

				{
					let (a_real, a_residual) = self.get_derivative(n1, &wrt, &input_map);
					let (b_real, b_residual) = self.get_derivative(n2, &wrt, &input_map);

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
				}

				// TODO: It should probably never happen, right?
				if wrt_this {
					(result, vec![1.0; residual.len()])
				} else {
					(result, residual)
				}
			},
			Operation::Unary(a_id, ref f, ref dfdx) => {
				let mut result = vec![0.0; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1];
				let mut residual = vec![0.0; result.len()];
				{	
					let (a_real, a_res) = self.get_derivative(a_id, &wrt, &input_map);
					for i in 0..a_real.len() {
						result[i] = f(a_real[i]);
						residual[i] = dfdx(a_real[i])*a_res[i];
					}
				}

				if wrt_this {
					(result, vec![1.0; residual.len()])
				} else {
					(result, residual)
				}
			},
			Operation::Binary(node_a_id, node_b_id, ref f, ref dfda, ref dfdb) => { // LHS, RHS, F, dF/dLHS, dF/dRHS
				// g( <a_real, a_res>, <b_real, b_res> ) = < g(a_real, b_real), dgda(a_real, b_real)*da + dgdb(a_real, b_real)*db>
				let mut result_real = vec![0.0; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1];
				let mut result_residual = vec![0.0; result_real.len()];
				{
					let (a_real, a_res) = self.get_derivative(node_a_id, &wrt, &input_map);
					let (b_real, b_res) = self.get_derivative(node_b_id, &wrt, &input_map);
					assert_eq!(a_real.len(), b_real.len());
					assert_eq!(a_res.len(), b_res.len());
					for i in 0..a_real.len() {
						result_real[i] = f(a_real[i], b_real[i]);
						result_residual[i] = dfda(a_real[i], b_real[i])*a_res[i] + dfdb(a_real[i], b_real[i])*b_res[i];
					}
				}

				if wrt_this {
					(result_real, vec![1.0; result_residual.len()])
				} else {
					(result_real, result_residual)
				}
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
		self.insert_op(node_id, Operation::Unary(
			node_id,
			Box::new(|x| { 1.0/x }),
			Box::new(|x| { -1.0/x.powf(2.0) }),
		))
	}

	fn constant_add(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::Unary(node_id, Box::new(move |x| { x+scalar }), Box::new(move |x| { 1.0 })))
	}

	fn constant_multiply(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::Unary(node_id, Box::new(move |x| { x*scalar }), Box::new(move |x| {scalar})))
	}

	fn power(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		self.insert_op(node_id, Operation::Unary(
			node_id,
			Box::new(move |x| { x.powf(scalar) }),
			Box::new(move |x| { scalar*x.powf(scalar-1.0) }),
		))
	}

	fn add(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId {
		self.insert_op(node_a_id, Operation::Binary(node_a_id, node_b_id, Box::new(|a, b| { a+b }), Box::new(|a, b| { 1.0 }), Box::new(|a, b| { 1.0 })))
	}

	fn hadamard_product(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId { // Schur/Element-wise product.
		assert_eq!(self.nodes[node_a_id].shape, self.nodes[node_b_id].shape);
		self.insert_op(node_a_id, Operation::Binary(
			node_a_id, node_b_id,
			Box::new(|a, b| { a*b }),
			Box::new(|a, b| { b }), // df/da
			Box::new(|a, b| { a }) // df/db
		))
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
	use super::{Graph, Dimension, Node, NodeId, apply_gradient};
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
		let d = g.add(b, c);
		// b = a+10  c = a*2  d = (a+10)+(a*2)
		
		let mut input = HashMap::new();
		input.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let (activations, gradients) = g.get_derivative(b, &[a], &input);
		// db/da (a + 10) = 1.0
		for dbda in gradients {
			println!("dB/dA: {}.  Expected: 1.0", dbda);
			assert!((dbda - 1.0).abs() < 0.001);
		}
		// dc/da = 2.0
		let (activations, gradients) = g.get_derivative(c, &[a], &input);
		for dcda in gradients {
			println!("dC/dA: {}.  Expected: 2.0", dcda);
			assert!((dcda - 2.0).abs() < 0.001);
		}
		let (activations, gradients) = g.get_derivative(d, &[a], &input);
		// dd/da = dd/da ( 2a + a + 10 ) = ( 3 )
		for ddda in gradients {
			println!("dD/dA: {}.  Expected: 3.0", ddda);
			assert!((ddda - 3.0).abs() < 0.001);
		}
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

		let (fx, dfdx) = g.get_derivative(abcdef, &[x], &input);
		for i in 0..10 {
			assert!((fx[i] - expected_result[i]).abs() <= EPSILON);
			assert!((dfdx[i] - expected_grad[i]).abs() <= EPSILON);
		}
	}

	#[test]
	fn numerical_gradient_check() {
		const DELTA : f32 = 0.1;
		const EPSILON : f32 = 0.01;
		let mut g = Graph::new();
		let x = g.input((1, 3));

		// This can be whatever method.
		let a = g.constant_add(x, 5.0);
		let b = g.constant_multiply(a, -1.5);
		let c = g.add(a, b);
		let d = g.sigmoid(c);
		let op = g.power(d, 2.0);

		let mut input = HashMap::new();
		for i in -256..256i32 {
			let j = (i as f32);
			let x0 = j-DELTA;
			let x1 = j;
			let x2 = j+DELTA;

			{
				input.insert(x, vec![x0, x1, x2]);
				let (out, outgrad) = g.get_derivative(op, &[x], &input);
				let numerical_grad = (out[2] - out[0]) / (x2 - x0);

				//println!("{:?} : {:?} {:?}", &[x0, x1, x2], &out, &outgrad);
				println!("f({}) -- Calculated gradient : {}.  Numerical approximation : {}.", j, outgrad[1], numerical_grad);
				assert!( (outgrad[1]-numerical_grad).abs() < EPSILON );
			}
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
		let mut activations : HashMap<usize, Vec<f32>> = HashMap::new();
		let mut gradients : HashMap<usize, Vec<f32>> = HashMap::new();
		let (out, dmm_dxg) = g.get_derivative(mm, &[x], &input);

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
	fn test_linear_regression() {
		const MULTIPLE : f32 = 3.0;
		let mut w_data = vec![1.0];

		let mut g = Graph::new();
		let x = g.input((1, 1));
		let w = g.input((1, 1));
		let y = g.matmul(x,w);

		let mut inputs = HashMap::new();
		for i in 0..200u32 {
			{
				inputs.insert(x, vec![i as f32]);	
				inputs.insert(w, w_data.clone());
				let (res, grad) = g.get_derivative(y, &[w], &inputs);
				let err = vec![MULTIPLE*(i as f32) - res[0]];
				println!("{} * {} = {} --- Error: {} --- Grad: {}", i, w_data[0], res[0], err[0], grad[0]);
				w_data[0] += 0.001 * err[0] * grad[0] * (i as f32);
			}
		}
		assert_eq!(MULTIPLE, w_data[0]);
	}

	#[test]
	fn test_deep_backprop() {
		let mut rng = rand::thread_rng();
		let mut g = Graph::new();

		let a = g.input((1, 1));
		let b = g.input((1, 1));
		let c = g.input((1, 1));
		let d = g.input((1, 1));

		let x = g.matmul(a, b);
		let y = g.matmul(x, c);
		let z = g.sigmoid(y);
		let w = g.matmul(z, d);

		// w = sigmoid((a*b)*c)*d
		// dw/dd = sigmoid((a*b)*c)
		// dw/da = b*c*d*e^(a*b*c) / (e^(a*b*c)+1)^2
		for i in 0..10000 {
			let a_ = rng.gen::<f32>()*10.0;
			let b_ = rng.gen::<f32>()*10.0;
			let c_ = rng.gen::<f32>()*10.0;
			let d_ = rng.gen::<f32>()*10.0;
			let expected_dw_dd = 1.0/(1.0 + (-1.0*a_*b_*c_).exp());
			let expected_dw_da = b_*c_*d_*(a_*b_*c_).exp() / (1.0+(a_*b_*c_).exp().powf(2.0));

			let mut inputs = HashMap::new();
			inputs.insert(a, vec![a_]);
			inputs.insert(b, vec![b_]);
			inputs.insert(c, vec![c_]);
			inputs.insert(d, vec![d_]);

			let (out, dw_dd) = g.get_derivative(w, &[d], &inputs);
			println!("Derivatives of w wrt d: {:?}.  Expected: {:?}", dw_dd, expected_dw_dd);
			assert!((dw_dd[0] - expected_dw_dd).abs() < 0.0001);
		}
	}

	#[test]
	fn test_backprop() {
		const INPUT_SIZE : usize = 2;
		const HIDDEN_SIZE : usize = 3;
		const OUTPUT_SIZE : usize = 1;

		// Dataset generator.
		let mut rng = rand::thread_rng();

		// Define graph
		let mut g = Graph::new();

		// Define inputs and variables.
		let mut learning_rate = 0.0001f32;
		let x = g.input((1, INPUT_SIZE));
		let w_ih = g.input((INPUT_SIZE, HIDDEN_SIZE));
		let h_bias = g.input((1, HIDDEN_SIZE));
		let w_ho = g.input((HIDDEN_SIZE, OUTPUT_SIZE));
		let y = g.input((1, OUTPUT_SIZE)); // For labels.

		let mut w_ih_data : Vec<f32> = (0u32..(INPUT_SIZE*HIDDEN_SIZE) as u32).map(|i| rng.gen::<f32>()*0.1).collect();
		let mut h_bias_data : Vec<f32> = vec![0.0; HIDDEN_SIZE];
		let mut w_ho_data : Vec<f32> = (0u32..(HIDDEN_SIZE*OUTPUT_SIZE) as u32).map(|i| rng.gen::<f32>()*0.1).collect();

		// Define operations.
		let hidden_z = g.matmul(x, w_ih);
		let hidden_z_biased = g.add(hidden_z, h_bias);
		let hidden_a = g.sigmoid(hidden_z_biased);
		let out = g.matmul(hidden_a, w_ho);

		// Define our errors.
		let out_inverse = g.constant_multiply(out, -1.0);
		let error_function = g.add(y, out_inverse);
		let cost = g.power(error_function, 2.0); // Squared error.

		let mut inputs = HashMap::new();
		for i in 0..10 {
			// Train an example.
			let x0 : bool = rng.gen();
			let x1 : bool = rng.gen();
			let example = vec![if x0 { 1.0 } else { 0.0 }, if x1 { 1.0 } else { 0.0 }];
			let label = vec![if x0 ^ x1 { 1.0 } else { 0.0 }];
			inputs.insert(x, example.clone());
			inputs.insert(w_ih, w_ih_data.clone());
			inputs.insert(h_bias, h_bias_data.clone());
			inputs.insert(w_ho, w_ho_data.clone());
			inputs.insert(y, label.clone());
			
			//let output : Vec<f32> = activations.get(&cost).unwrap();
			//assert!(!output[0].is_nan());
			// fn apply_gradient(weight_matrix : &mut Vec<f32>, input : &Vec<f32>, gradient : &Vec<f32>, error : &Vec<f32>, learning_rate : f32)
			{
				let res = g.get_output(out, &inputs);
				println!("Output: {:?}.  Goal: {:?}", res, label);
				let (res, dcost_dw) = g.get_derivative(cost, &[w_ho], &inputs); // d error wrt w_ih
				println!("Squared error: {:?}.  dCost/dWho : {:?}", res, dcost_dw);
				let (res, dcost_dw) = g.get_derivative(cost, &[w_ih], &inputs); 
				println!("Squared error: {:?}.  dCost/dWih : {:?}", res, dcost_dw);
			}
			//apply_gradient(&mut w_ho_data, activations.get(&hidden_a).unwrap(), gradients.get(&w_ho).unwrap(), &err, learning_rate);

			if (i+1) % 1000 == 0 {
				learning_rate *= 0.1;
			}
		}
		// delta_w_ij = alpha * (tj - yj) * g'(h_j) * x_i
		// dw = learning_rate * (truth - out) * derivative_of_activation(input_activation) * input_activation

		// Verify network.
		inputs.insert(w_ih, w_ih_data.clone());
		inputs.insert(w_ho, w_ho_data.clone());

		inputs.insert(x, vec![0f32, 0.0]);
		let res = g.get_output(out, &inputs);
		println!("{} ^ {} : {}", 0, 0, res[0]);
		//assert!(res[0] < 0.1);

		inputs.insert(x, vec![0f32, 1.0]);
		let res = g.get_output(out, &inputs);
		println!("{} ^ {} : {}", 0, 1, res[0]);
		//assert!(res[0] > 0.9);

		inputs.insert(x, vec![1.00f32, 0.0]);
		let res = g.get_output(out, &inputs);
		println!("{} ^ {} : {}", 1, 0, res[0]);
		//assert!(res[0] > 0.9);

		inputs.insert(x, vec![1.0f32, 1.0]);
		let res = g.get_output(out, &inputs);
		println!("{} ^ {} : {}", 1, 1, res[0]);
		//assert!(res[0] < 0.1);
		assert!(false);
	}
}
