#ifndef BIT_VECTOR_GLSL
#define BIT_VECTOR_GLSL

#define bit_vector_get(buffer, node) (((buffer)[(node) / 16] & (1u << ((node) % 16))) > 0)

#define bit_vector_set(buffer, node) ((atomicOr((buffer)[(node) / 16], 1u << ((node) % 16)) & (1u << ((node) % 16))) > 0)

#endif
