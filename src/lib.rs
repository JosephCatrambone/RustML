
/***
 * Standard reverse-mode automatic differentiation system.
 * Uses backwards diff to compute derivative WRT some variable at a given interval.
 */

use std::f32;
use std::vec;
use std::collections::HashMap;
use std::collections::hash_set::HashSet;

extern crate ocl;
use ocl::{ProQue, Buffer};

static KERNEL_SOURCE: &'static str = include_str!("kernel_source.cl");

type Dimension = (usize, usize);
type NodeId = usize;

//		ActivationFunction::Sigmoid => 1.0f32/(1.0 + (-value).exp()),

/*
fn unary_operation(a : &Vec<f32>, c : &mut Vec<f32>, op : Box<Fn(f32)->f32>) {
	for i in 0..c.len() {
		c[i] = op(a);
	}
}

let foo = Box::new(|x| { x*2 })
*/

enum Operation {
	Input,
	MatrixMultiply(NodeId, NodeId),
	ConstantAdd(NodeId, f32),
	ConstantMultiply(NodeId, f32),
	ElementAdd(NodeId, NodeId),
	ElementMultiply(NodeId, NodeId),
	ElementInverse(NodeId),
	Sigmoid(NodeId),
	// TODO: Find out if elementInverse(constantAdd(1, elementExp(elementMultiply(-1, input)))) == d/dx sigmoid
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
			Operation::Sigmoid(node) => {
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = 1.0/(1.0+(-a[i]).exp());
				}
				result
			}
		}
	}

	// TODO: Understand Tann's slides + Torch's updateOutput, updateGradInput, and accGradParameters

	fn get_gradient(&self, node_id : NodeId, wrt : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		match self.nodes[node_id].operation {
			Operation::Input => {
				vec![
					if node_id == wrt {
						1.0
					} else {
						0.0
					} ; self.nodes[node_id].shape.0*self.nodes[node_id].shape.1
				]	
			},
			Operation::MatrixMultiply(n1, n2) => {
				vec![] // TODO
			},
			Operation::ConstantAdd(node, scalar) => {
				let vec_len = self.nodes[node_id].shape.0*self.nodes[node_id].shape.1;
				// d/dx c*x = c
				// d/dx c*y = 0
				// TODO: Raise exception if someone passes wrt as node_id, because d/dx f(+) makes no sense.
				if wrt == node {
					vec![scalar; vec_len]
				} else {
					vec![0.0; vec_len]
				}
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
				// TODO
				let a : Vec<f32> = self.get_output(node_a_id, &input_map);
				let b : Vec<f32> = self.get_output(node_b_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]+b[i];
				}
				result
			},
			Operation::ElementMultiply(node_a_id, node_b_id) => {
				// TODO
				let a : Vec<f32> = self.get_output(node_a_id, &input_map);
				let b : Vec<f32> = self.get_output(node_b_id, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = a[i]*b[i];
				}
				result
			},
			Operation::ElementInverse(node) => {
				// TODO
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = 1.0/a[i];
				}
				result
			},
			Operation::Sigmoid(node) => {
				// TODO:
				let a : Vec<f32> = self.get_output(node, &input_map);
				let mut result = vec![0.0; a.len()];
				for i in 0..a.len() {
					result[i] = 1.0/(1.0+(-a[i]).exp());
				}
				result
			}
		}
	}

	// TODO:
	//fn get_output_shape(&self, node_id : NodeId) {}

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

	fn hadamard_product(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId { // Schur/Element-wise product.
		assert_eq!(self.nodes[node_a_id].shape, self.nodes[node_b_id].shape);
		self.insert_op(node_a_id, Operation::ElementMultiply(node_a_id, node_b_id))
	}
}

#[cfg(test)]
mod tests {
	use super::{Graph, Dimension, Node, NodeId};
	use std::collections::HashMap;

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
		
		let mut input = HashMap::new();
		input.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let res = g.get_output(b, &input);
		let dres_da = g.get_gradient(b, a, &input);
		let dres_db = g.get_gradient(b, b, &input);
		assert_eq!(dres_da[0], 10.0);
		assert_eq!(dres_db[0], 0.0);
	}

	#[test]
	fn test_backprop() {
		let mut g = Graph::new();
		let x = g.input((1, 10));
	}

    #[test]
    fn integration_full_test() {
		let g = Graph::new();
    }
}
