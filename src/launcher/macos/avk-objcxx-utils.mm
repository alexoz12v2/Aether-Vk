#include "avk-objcxx-utils.h"

#import <Foundation/Foundation.h>

#include <string>

namespace avk {

std::filesystem::path getResourcePath() {
  NSString* nsPath = [[NSBundle mainBundle] resourcePath];
  // convert to std::string assuming UTF-8 encoding
  std::string pathStr = [nsPath UTF8String];
  return std::filesystem::path(pathStr);
}

}  // namespace avk