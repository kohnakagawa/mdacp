//----------------------------------------------------------------------
#include <iostream>
#include <map>
#include <fstream>
#include <stdlib.h>
#include "mpistream.h"
#include "parameter.h"
using namespace std;
//----------------------------------------------------------------------
Parameter::Parameter(const char *filename) {
  valid = true;
  ifstream is(filename);
  if (is.fail()) {
    mout << "Could not open input file " << filename << std::endl;
    valid = false;
  }
  ReadFromStream(is);
}
//----------------------------------------------------------------------
Parameter::Parameter(istream &is) {
  valid = true;
  ReadFromStream(is);
}
//----------------------------------------------------------------------
void
Parameter::LoadFromFile(const char* filename) {
  ifstream is(filename);
  if (is.fail()) {
    mout << "Could not open input file " << filename << std::endl;
    valid = false;
  }
  ReadFromStream(is);
}
//----------------------------------------------------------------------
void
Parameter::ReadFromStream(istream &is) {
  string line;
  while (getline(is, line)) {
    size_t index = line.find("=");
    if (string::npos != index) {
      string key = line.substr(0, index);
      string value = line.substr(index + 1, line.length());
      params.insert(ptype::value_type(key, value));
    }
  }
};
//----------------------------------------------------------------------
bool
Parameter::Contains(string key) {
  if (params.find(key) == params.end()) {
    return false;
  } else {
    return true;
  }
}
//----------------------------------------------------------------------
string
Parameter::GetString(string key) {
  return GetStringDef(key, "");
}
//----------------------------------------------------------------------
string
Parameter::GetStringDef(string key, string value) {
  if (!Contains(key)) {
    return value;
  }
  return params[key];
}
//----------------------------------------------------------------------
double
Parameter::GetDouble(string key) {
  return GetDoubleDef(key, 0.0);
}
//----------------------------------------------------------------------
double
Parameter::GetDoubleDef(string key, double value) {
  if (!Contains(key)) {
    return value;
  }
  return atof(params[key].c_str());
}
//----------------------------------------------------------------------
int
Parameter::GetInteger(string key) {
  return GetIntegerDef(key, 0);
}
//----------------------------------------------------------------------
int
Parameter::GetIntegerDef(string key, int value) {
  if (!Contains(key)) {
    return value;
  }
  return atoi(params[key].c_str());
}
//----------------------------------------------------------------------
bool
Parameter::GetBoolean(string key) {
  return GetBooleanDef(key, false);
}
//----------------------------------------------------------------------
bool
Parameter::GetBooleanDef(string key, bool value) {
  if (!Contains(key)) {
    return value;
  }
  if ("yes" == params[key] || "Yes" == params[key]) {
    return true;
  } else {
    return false;
  }
}
//----------------------------------------------------------------------
void
Parameter::SetDouble(string key, double value) {
  char s[256];
  sprintf(s, "%f", value);
  params[key] = s;
}
//----------------------------------------------------------------------
void
Parameter::ShowAll(void) {
  ptype::iterator it = params.begin();
  while (it != params.end()) {
    mout << (*it).first << ":" << (*it).second << std::endl;
    ++it;
  }
}
//----------------------------------------------------------------------
