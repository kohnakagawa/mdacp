#include <stdio.h>
#include <stdarg.h>
#include "mdconfig.h"

//----------------------------------------------------------------------
const char *MDACP_VERSION = "MDACP Ver 2.20";
//----------------------------------------------------------------------
const char *Direction::name_str[MAX_DIR] = {"left", "right", "back", "forward", "down", "up"};
//----------------------------------------------------------------------
void
debug_printf(const char *format, ...) {
#ifdef DEBUG
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
#endif
}
//----------------------------------------------------------------------
