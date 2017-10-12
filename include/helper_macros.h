//----------------------------------------------------------------------
#ifndef helper_macros_h
#define helper_macros_h
//----------------------------------------------------------------------
#define MDACP_CONCATENATE(x, y) x ## y
#define MDACP_CONCAT(x, y) MDACP_CONCATENATE(x, y)
#define MDACP_EMPTY
#define MDACP_NAMESPACE_AT(n, f) MDACP_CONCAT(n:, :f)
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
