
/***
 * Standard forward and reverse-mode automatic differentiation system.
 * Uses backwards diff to compute gradients and forward mode to compute derivatives.
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
	//MatrixMultiplyTranspose(NodeId, NodeId), // Second item is the transpose.  More efficient than matmul(a, transpose(b))
	Transpose(NodeId),
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
	fn get_shape(&self, node_id : NodeId) -> Dimension {
		self.nodes[node_id].shape.clone()
	}

	fn get_output(&self, node_id : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		let mut cache = HashMap::new();
		self.get_output_with_cache(node_id, &input_map, &mut cache)
	}

	fn get_forward(&self, node_id : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> HashMap<NodeId, Vec<f32>> {
		let mut cache = HashMap::new();
		self.get_output_with_cache(node_id, &input_map, &mut cache);
		cache
	}

	fn get_output_with_cache(&self, node_id : NodeId, input_map : &HashMap<NodeId, Vec<f32>>, mut cached_activation : &mut HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		if cached_activation.contains_key(&node_id) {
			return cached_activation.get(&node_id).unwrap().clone();
		}

		let new_cached_activation = match self.nodes[node_id].operation {
			Operation::Input => { 
				let data_vec = input_map.get(&node_id).expect(&format!("Input Map is missing value for input node {}", node_id)).clone();
				assert_eq!(data_vec.len(), self.nodes[node_id].shape.0*self.nodes[node_id].shape.1);
				data_vec
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

				//let mut result = vec![0.0; a_height*b_width];
				let mut result = vec![0.0; self.nodes[n1].shape.0*self.nodes[n2].shape.1];

				// Set up the buffer for this node.
				let a : Vec<f32> = self.get_output_with_cache(n1, &input_map, &mut cached_activation);
				let b : Vec<f32> = self.get_output_with_cache(n2, &input_map, &mut cached_activation);

				// Multiply
				for i in 0..a_height { // Column [Iterating over row]
					for k in 0..b_width { // Row/Width [Iterating over column]
						let mut accumulator = 0.0;
						for j in 0..a_width { // Column
							assert!(j + i*a_width < a.len());
							assert!(k + j*b_width < b.len());
							accumulator += a[j + i*a_width]*b[k + j*b_width];
						}
						result[k + i*c_width] = accumulator;
					}
				}

				result
			},
			Operation::Transpose(n) => {
				// Verify shapes
				let a_width = self.nodes[n].shape.1; // Columns (j)
				let a_height = self.nodes[n].shape.0; // Rows (i)

				let mut result = vec![0.0; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1];

				// Set up the buffer for this node.
				let a : Vec<f32> = self.get_output_with_cache(n, &input_map, &mut cached_activation);

				// Multiply
				for i in 0..a_height { // Column [Iterating over row]
					for j in 0..a_width { // Row/Width [Iterating over column]
						result[i + j*a_width] = a[j + i*a_width];
					}
				}

				result
			},
			Operation::Unary(node_id, ref f, _) => {
				let a : Vec<f32> = self.get_output_with_cache(node_id, &input_map, &mut cached_activation);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = f(a[i]);
				}
				result
			},
			Operation::Binary(node_a_id, node_b_id, ref f, _, _) => { // LHS, RHS, F, dF/dLHS, dF/dRHS
				let a : Vec<f32> = self.get_output_with_cache(node_a_id, &input_map, &mut cached_activation);
				let b : Vec<f32> = self.get_output_with_cache(node_b_id, &input_map, &mut cached_activation);
				assert_eq!(a.len(), b.len());
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = f(a[i], b[i]);
				}
				result
			},
		};

		cached_activation.insert(node_id, new_cached_activation.clone());

		new_cached_activation
	}

	fn get_gradient(&self, node_id : NodeId, wrt : &[NodeId], forward_pass : &HashMap<NodeId, Vec<f32>>) -> HashMap<(NodeId, NodeId), Vec<f32>> {
		// Get the derivative of node_id with respect to ALL the inputs.
		let mut grad = HashMap::new();
		for &v in wrt {
			self.get_gradient_with_cache(node_id, v, &forward_pass, &mut grad);
		}
		grad
	}

	fn get_gradient_with_cache(&self, node_id : NodeId, wrt : NodeId, forward_pass : &HashMap<NodeId, Vec<f32>>, mut grads : &mut HashMap<(NodeId, NodeId), Vec<f32>>) {
		// y = f(u) u = g(x).  dy/dx = dy/du * du/dx.  dy/du = f'(u) du/dx = g'(x)
		// y = f(u, v), u=g(a, b), v=h(a, b).  dy/da = dy/du*du/da + dy/dv*dv/da
		
		// Currently seeking dNodeID/dwrt
		if grads.contains_key(&(node_id, wrt)) {
			return;
		}

		//println!("DEBUG: get_derivative({:?}, {:?}, {:?}, {:?}, {:?})", node_id, wrt, input_map, activations, gradients);

		match self.nodes[node_id].operation {
			Operation::Input => {
				//let len = forward_pass.get(&node_id).unwrap().len(); 
				let len = self.nodes[node_id].shape.0 * self.nodes[node_id].shape.1;
				if node_id == wrt { // dx/dx = 1.
					if self.nodes[node_id].shape.0 == 1 {
						// Derivative of a vector dx should be all ones.
						grads.insert((node_id, wrt), vec![1.0; len]);
					} else {
						// Derivative of a MATRIX c should be identity.
						let mut data = vec![];
						for r in 0..self.nodes[node_id].shape.0 {
							for c in 0..self.nodes[node_id].shape.1 {
								if r == c {
									data.push(1.0);
								} else {
									data.push(0.0);
								}
							}
						}
						grads.insert((node_id, wrt), data);
					}
				} else {
					grads.insert((node_id, wrt), vec![0.0; len]);
				}
			},
			Operation::MatrixMultiply(n1, n2) => {
				// this = node_id, 
				// this = f(x, y).  df/dn -> dthis/dn = dthis/dx*dx/dn + dthis/dy*dy/dn
				// dThis/dT = dThis/dn1*dn1/dT + dThis/dn2*dn2/dT

				// Verify shapes
				let a_width = self.nodes[n1].shape.1; // Columns (j)
				let a_height = self.nodes[n1].shape.0; // Rows (i)
				let b_width = self.nodes[n2].shape.1; // Columns (k)
				let b_height = self.nodes[n2].shape.0; // Rows (j)
				let c_width = b_width;
				let c_height = a_height;
				assert_eq!(a_width, b_height);

				// How does this change wrt n1?  Each element changes at a rate n2.
				// How does this change wrt n2?  Each changes at a rate n1.
				// dThis/dn1 * dn1/dt = matmul(n2, grads((n1, nT))) + matmul(n1, grads(n2, nT))
				self.get_gradient_with_cache(n1, wrt, &forward_pass, &mut grads);
				self.get_gradient_with_cache(n2, wrt, &forward_pass, &mut grads); // TODO: Can these be mutually recursive?
				let grad_n1 = grads.get(&(n1, wrt)).unwrap().clone();
				let grad_n2 = grads.get(&(n2, wrt)).unwrap().clone();

				let act_n1 = forward_pass.get(&n1).unwrap();
				let act_n2 = forward_pass.get(&n2).unwrap();

				let mut grad = vec![0.0; self.nodes[n1].shape.0*self.nodes[n2].shape.1];

				{
					// Multiply
					for i in 0..a_height { // Column [Iterating over row]
						for k in 0..b_width { // Row/Width [Iterating over column]
							let mut accumulator = 0.0;
							for j in 0..a_width { // Column
								accumulator += (act_n1[j + i*a_width]*grad_n2[k + j*b_width]) + (grad_n1[j + i*a_width]*act_n2[k + j*b_width]);
							}
							grad[k + i*c_width] = accumulator;
						}
					}

					grads.insert((node_id, wrt), grad);
				}

			},
			Operation::Transpose(n) => {
				// Verify shapes
				let a_width = self.nodes[n].shape.1; // Columns (j)
				let a_height = self.nodes[n].shape.0; // Rows (i)

				self.get_gradient_with_cache(n, wrt, &forward_pass, &mut grads);
				let grad = grads.get(&(n, wrt)).unwrap().clone(); // TODO: Can we avoid a copy here?
				let mut result = vec![0.0; self.nodes[n].shape.0*self.nodes[n].shape.1];

				// Multiply
				for i in 0..a_height { // Column [Iterating over row]
					for j in 0..a_width { // Row/Width [Iterating over column]
						result[i + j*a_width] = grad[j + i*a_width]; // Nope.  Not (x).T' * x'.  Just x'.T.
					}
				}

				grads.insert((node_id, wrt), result);
			},
			Operation::Unary(n, ref f, ref df) => {
				self.get_gradient_with_cache(n, wrt, &forward_pass, &mut grads);
				let grad = grads.get(&(n,wrt)).unwrap().clone();
				let act = forward_pass.get(&n).unwrap();
				let mut result = vec![0.0; act.len()];
				for i in 0..act.len() {
					result[i] = df(act[i])*grad[i];
				}
				grads.insert((node_id,wrt), result);
			},
			Operation::Binary(node_a_id, node_b_id, ref f, ref df_wrt_a, ref df_wrt_b) => { // LHS, RHS, F, dF/dLHS, dF/dRHS
				// z = f(x,y)
				// x = x(u,v)
				// y = y(u,v)
				// dz/du = dz/dx*dx/du + dz/dy*dy/du
				// dNode/wrt = dNode/dNodeA * dNodeA/dwrt + dNode/dNodeN * dNodeB/dwrt;
				// dNode/dNodeA = df_wrt_a(A)
				// dNodeA/dy = get recursive.
				self.get_gradient_with_cache(node_a_id, wrt, &forward_pass, &mut grads);
				self.get_gradient_with_cache(node_b_id, wrt, &forward_pass, &mut grads);
				let grad_a = grads.get(&(node_a_id, wrt)).unwrap().clone(); 
				let grad_b = grads.get(&(node_b_id, wrt)).unwrap().clone();
				let act_a = forward_pass.get(&node_a_id).unwrap(); 
				let act_b = forward_pass.get(&node_b_id).unwrap();
				assert_eq!(act_a.len(), act_b.len());
				let mut result = vec![0.0; act_a.len()];
				for i in 0..act_a.len() {
					result[i] = df_wrt_a(act_a[i], act_b[i])*grad_a[i] + df_wrt_b(act_a[i], act_b[i])*grad_b[i];
				}
				grads.insert((node_id, wrt), result);
			},
		};
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

	fn insert_op(&mut self, shape : Dimension, op : Operation) -> NodeId {
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
		let shape = self.get_shape(node_id);
		self.insert_op(shape, Operation::Unary(
			node_id,
			Box::new(|x| { 1.0/x }),
			Box::new(|x| { -1.0/x.powf(2.0) }),
		))
	}

	fn constant_add(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		let shape = self.get_shape(node_id);
		self.insert_op(shape, Operation::Unary(node_id, Box::new(move |x| { x+scalar }), Box::new(move |x| { 1.0 })))
	}

	fn constant_multiply(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		let shape = self.get_shape(node_id);
		self.insert_op(shape, Operation::Unary(node_id, Box::new(move |x| { x*scalar }), Box::new(move |x| {scalar})))
	}

	fn power(&mut self, node_id : NodeId, scalar : f32) -> NodeId {
		let shape = self.get_shape(node_id);
		self.insert_op(shape, Operation::Unary(
			node_id,
			Box::new(move |x| { x.powf(scalar) }),
			Box::new(move |x| { scalar*x.powf(scalar-1.0) }),
		))
	}

	fn add(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId {
		let shape = self.get_shape(node_a_id);
		self.insert_op(shape, Operation::Binary(node_a_id, node_b_id, Box::new(|a, b| { a+b }), Box::new(|a, b| { 1.0 }), Box::new(|a, b| { 1.0 })))
	}

	fn hadamard_product(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId { // Schur/Element-wise product.
		let shape = self.get_shape(node_a_id);
		assert_eq!(self.nodes[node_a_id].shape, self.nodes[node_b_id].shape);
		self.insert_op(shape, Operation::Binary(
			node_a_id, node_b_id,
			Box::new(|a, b| { a*b }),
			Box::new(|a, b| { b }), // df/da
			Box::new(|a, b| { a }) // df/db
		))
	}

	fn sigmoid(&mut self, node_id : NodeId) -> NodeId {
		let shape = self.get_shape(node_id);
		self.insert_op(shape, Operation::Unary(
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

		let forward = g.get_forward(abcdef, &input);
		let grad = g.get_gradient(abcdef, &[x], &forward);
		for i in 0..10 {
			assert!((grad.get(&(abcdef, x)).unwrap()[i] - expected_grad[i]).abs() <= EPSILON);
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
				let out = g.get_output(op, &input);
				let forward = g.get_forward(op, &input);
				let grad = g.get_gradient(op, &[x], &forward);
				let numerical_grad = (out[2] - out[0]) / (x2 - x0);

				//println!("{:?} : {:?} {:?}", x0, &out, &outgrad);
				println!("f({}) -- Calculated gradient : {}.  Numerical approximation : {}.", j, grad.get(&(op,x)).unwrap()[0], numerical_grad);
				assert!( (numerical_grad-grad.get(&(op,x)).unwrap()[0]).abs() < EPSILON );
			}
		}
	}

	#[test]
	fn test_matrix_multiply_gradient() {
		let mut g = Graph::new();
		let x = g.input((4, 4));
		let y = g.input((4, 5));

		let mut input : HashMap<usize, Vec<f32>> = HashMap::new();
		let mm = g.matmul(x, y);
		let x_data: Vec<f32> = (0..16u32).map(|i| i as f32).collect(); // Get rid of the map for just an array of ints.
		let y_data: Vec<f32> = (0..20u32).map(|i| { 8.0 - i as f32 } ).collect(); 
		
		input.insert(x, x_data.clone());
		input.insert(y, y_data.clone());

		let mut forward = HashMap::new();
		g.get_output_with_cache(mm, &input, &mut forward);

		let mut gradients : HashMap<(NodeId, NodeId), Vec<f32>> = g.get_gradient(mm, &[x], &forward);
		let dxy_wrt_x = gradients.get(&(mm, x)).unwrap();

		println!("X:{:?}", x_data);
		println!("Y:{:?}", y_data);
		println!("d x*y wrt X:{:?}", dxy_wrt_x);
		assert!(&dxy_wrt_x[..] == &y_data[..]);
	}

	#[test]
	fn test_linear_regression() {
		const MULTIPLE : f32 = 3.0;
		let mut w_data = vec![1.0];

		let mut g = Graph::new();
		let x = g.input((1, 1));
		let w = g.input((1, 1));
		let out = g.matmul(x,w);
		let y = g.input((1, 1));
		let difference = g.constant_multiply(out, -1.0);
		let error = g.add(y, difference);
		let cost = g.power(error, 2.0);

		let mut inputs = HashMap::new();
		for i in 0..200u32 {
			{
				inputs.insert(x, vec![i as f32]);	
				inputs.insert(w, w_data.clone());
				inputs.insert(y, vec![MULTIPLE*(i as f32)]);
				let forward = g.get_forward(cost, &inputs);
				let grad = g.get_gradient(cost, &[w], &forward);
				let err = vec![MULTIPLE*(i as f32) - forward.get(&out).unwrap()[0]];
				println!("{} --- Error: {} --- Grad: {:?}", i, err[0], grad);
				w_data[0] -= 0.001 * grad.get(&(cost, w)).unwrap()[0];
			}
		}
		assert_eq!(MULTIPLE, w_data[0]);
	}

	#[test]
	fn test_matrix_gradient_sizes() {
		let mut g = Graph::new();

		let a = g.input((1, 3));
		let w1 = g.input((3, 4));
		let w2 = g.input((4, 2));
		let w3 = g.input((2, 5));

		let b = g.matmul(a, w1);
		let c = g.matmul(b, w2);
		let d = g.matmul(c, w3);

		let mut inputs : HashMap<usize, Vec<f32>> = HashMap::new();
		inputs.insert(a, vec![1.0f32, 2.0, 3.0]);
		let arr : Vec<f32> = (0..12u32).map(|x| { x as f32 }).collect();
		inputs.insert(w1, arr);
		let arr : Vec<f32> = (0..8u32).map(|x| { x as f32 }).collect();
		inputs.insert(w2, arr);
		let arr : Vec<f32> = (0..10u32).map(|x| { x as f32 }).collect();
		inputs.insert(w3, arr);

		let mut forward = HashMap::new();
		g.get_output_with_cache(d, &inputs, &mut forward);
		let gradients = g.get_gradient(d, &[w1, w2, w3], &forward);

		println!("Grad sizes: {:?}", gradients);
		/*
//		assert!(false);
		a -> 1x3
		w1 -> 3x4
		w2 -> 4x2
		w3 -> 2x5

		b -> 1x4
		c -> 1x2
		d -> 1x5

		{
		(a, w3): [0, 0, 0], 
		(b, w3): [0, 0, 0, 0], 
		(c, w1): [16, 22], 
		(w3, w2): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		(w3, w1): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		(w2, w2): [1, 0, 0, 1, 0, 0, 0, 0], 
		(w1, w1): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
		(a, w2): [0, 0, 0], 
		(w3, w3): [1, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
		(w2, w1): [0, 0, 0, 0, 0, 0, 0, 0], 
		(d, w3): [552, 716, 0, 0, 0], 
		(b, w2): [0, 0, 0, 0], 
		(b, w1): [1, 2, 3, 0], 
		(w2, w3): [0, 0, 0, 0, 0, 0, 0, 0], 
		(d, w1): [110, 148, 186, 224, 262], 
		(w1, w2): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		(d, w2): [190, 260, 330, 400, 470], 
		(c, w3): [0, 0], 
		(a, w1): [0, 0, 0], 
		(c, w2): [32, 38], 
		(w1, w3): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		}
		*/
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
			
			let mut forward = HashMap::new();
			g.get_output_with_cache(cost, &inputs, &mut forward);
			let gradients = g.get_gradient(cost, &[w_ih, w_ho], &forward);
			let dwih = gradients.get(&(cost, w_ih)).unwrap().clone();
			let dwho = gradients.get(&(cost, w_ho)).unwrap().clone();
			println!("CostNode: {}.  w_ihNode: {}.  w_hoNode: {}", cost, w_ih, w_ho);
			println!("Grads: {:?}", gradients);
			println!("dWih: {:?}", dwih);
			println!("Wih: {:?}", w_ih_data);
			println!("dWho: {:?}", dwho);
			println!("Who: {:?}", w_ho_data);
			// Apply the gradients.
			for i in 0..w_ih_data.len() {
				w_ih_data[i] += learning_rate * dwih[i];
			}
			for i in 0..w_ho_data.len() {
				w_ho_data[i] += learning_rate * dwho[i];
			}

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
