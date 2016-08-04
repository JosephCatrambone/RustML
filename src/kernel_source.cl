
__kernel void matrix_element_multiply(__global float * buffer, float scalar) {
	buffer[get_global_id(0)] *= scalar;
}
