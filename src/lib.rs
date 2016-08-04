
use std::vec;
use std::collections::HashMap;
use std::collections::hash_set::HashSet;

extern crate ocl;
use ocl::{ProQue, Buffer};

static KERNEL_SOURCE: &'static str = include_str!("kernel_source.cl");

type Dimension = (usize, usize);
type NodeId = usize;

#[derive(Hash, Eq, PartialEq, Debug)]
enum Operation {
	Input,
	MatrixMul(NodeId, NodeId),
	Sigmoid(NodeId),
}

struct Node {
	id : NodeId,
	input_nodes : Vec<NodeId>,
	input_shape : Option<Dimension>,
	output_shape : Option<Dimension>,
	operation : Operation,
	buffer : Buffer<f32>,
}

struct Graph {
	proque : ProQue,
	nodes : Vec<Node>,
}

impl Graph {
	fn new() -> Graph {
		Graph {
			proque : ProQue::builder().src(KERNEL_SOURCE).dims([3]).build().unwrap(),
			nodes : vec![],
		}
	}

	fn get_output(&self, node_id : NodeId, input_map : HashMap<NodeId, Vec<f32>>) -> Vec<f32> {
		//(node.operation)(&self, input_map)
		vec![]
	}

	fn new_input(&mut self, shape : Dimension) -> NodeId {
		let mut n = Node {
			id : 0,
			input_shape : Some(shape),
			output_shape : Some(shape),
			input_nodes : vec![],
			operation : Operation::Input,
			buffer : self.proque.create_buffer::<f32>().unwrap()
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
		let mut input = HashMap::new();
		let mut g = Graph::new();
		let n : NodeId = g.new_input((10, 11));
		input.insert(n, vec![0.0, 1.0, 2.0]);
		g.get_output(n, input);
	}

    #[test]
    fn integration_full_test() {
		let g = Graph::new();
    }
}
