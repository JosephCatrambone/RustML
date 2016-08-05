
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

#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
enum Operation<F> {
	Input,
	MatrixMultiply(NodeId, NodeId),
	MatrixBinaryOp(NodeId, NodeId, F),
	MatrixUnaryOp(NodeId, F),
//	Transpose(NodeId),
}

struct Node<F> {
	id : NodeId,
	shape : Dimension,
	operation : Operation<F>,
	buffer : Vec<f32>,
}

impl<F> Node<F> {
	fn get_value(&self, x : usize, y : usize) -> f32 {
		self.buffer[x + y*self.shape.1]
	}

	fn set_value(&mut self, x : usize, y : usize, value : f32) {
		self.buffer[x + y*self.shape.1] = value;
	}

	fn get_rows(&self) -> usize {
		self.shape.0
	}

	fn get_columns(&self) -> usize {
		self.shape.1
	}

	fn get_width(&self) -> usize {
		self.shape.1
	}

	fn get_height(&self) -> usize {
		self.shape.0
	}
}

struct Graph<F> {
//	proque : ProQue,
	nodes : Vec<Node<F>>,
}

impl<F> Graph<F> where F: Fn(f32, f32)->f32 {
	fn new() -> Graph<F> {
		Graph {
//			proque : ProQue::builder().src(KERNEL_SOURCE).dims([3]).build().unwrap(),
			nodes : vec![],
		}
	}

	// Graph methods
	fn get_output(&mut self, node_id : NodeId, input_map : &HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		//(node.operation)(&self, input_map)
		match self.nodes[node_id].operation {
			Operation::Input => input_map.get(&node_id).unwrap().clone(),
			Operation::MatrixMultiply(n1, n2) => {
				// Verify shapes
				let a_width = self.nodes[n1].shape.1; // Columns (j)
				let a_height = self.nodes[n1].shape.0; // Rows (i)
				let b_width = self.nodes[n2].shape.1; // Columns (k)
				let b_height = self.nodes[n2].shape.0; // Rows (j)
				let c_width = b_width;
				let c_height = a_height;
				assert_eq!(a_width, b_height);
				assert_eq!(a_height*b_width, self.nodes[node_id].buffer.len());

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
						self.nodes[node_id].buffer[k + i*c_width] = accumulator;
					}
				}

				self.nodes[node_id].buffer.clone()
			},
			Operation::MatrixBinaryOp(n1, n2, f) => {
				let a : Vec<f32> = self.get_output(n1, &input_map);
				let b : Vec<f32> = self.get_output(n2, &input_map);

				let vec_len = self.nodes[node_id].buffer.len();			

				for i in 0..vec_len {
					self.nodes[node_id].buffer[i] = f(a[i], b[i]);
				}

				self.nodes[node_id].buffer.clone()
			},
			Operation::MatrixUnaryOp(n1, f) => {
				let a : Vec<f32> = self.get_output(n1, &input_map);
				let vec_len = self.nodes[node_id].buffer.len();			
			
				for i in 0..vec_len {
					self.nodes[node_id].buffer[i] = f(a[i], i as f32);
				}

				self.nodes[node_id].buffer.clone()
			}
		}
	}

	// Node creation
	fn new_input(&mut self, shape : Dimension) -> NodeId {
		let mut n = Node {
			id : 0,
			shape : shape,
			operation : Operation::Input,
			buffer : vec![0.0; shape.0*shape.1], //self.proque.create_buffer::<f32>().unwrap()
		};
		self.nodes.push(n);
		let id = self.nodes.len()-1;
		n.id = id;
		id
	}

	fn new_matmul(&mut self, node_a_id : NodeId, node_b_id : NodeId) -> NodeId {
		let mut n = Node {
			id : 0,
			shape : (self.nodes[node_a_id].shape.0, self.nodes[node_b_id].shape.1),
			operation : Operation::MatrixMultiply(node_a_id, node_b_id),
			buffer : vec![0.0; self.nodes[node_a_id].shape.0*self.nodes[node_b_id].shape.1],
		};
		self.nodes.push(n);
		let id = self.nodes.len()-1;
		n.id = id;
		id
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
		let m : NodeId = g.new_input((3, 3));
		let n : NodeId = g.new_input((1, 3));
		let o = g.new_matmul(n, m);
		let j : NodeId = g.new_input((3, 5));
		let k = g.new_matmul(m, j);

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
    fn integration_full_test() {
		let g = Graph::new();
    }
}
