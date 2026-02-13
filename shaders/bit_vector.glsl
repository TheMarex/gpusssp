#ifndef BIT_VECTOR_GLSL
#define BIT_VECTOR_GLSL

#define bit_vector_get(buffer, node) (((buffer)[(node) / 32] & (1u << ((node) % 32))) > 0)

#define bit_vector_set(buffer, node) ((atomicOr((buffer)[(node) / 32], 1u << ((node) % 32)) & (1u << ((node) % 32))) > 0)

#endif
