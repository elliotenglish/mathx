#print("@platforms")

#INCLUDE_PREFIX=select({
#    "@platforms//os:macos":"local/include/",
#    "@platforms//os:linux":"local"})
#vtk_version=select({
#  "os:macos":"9.2",
#  "os:linux":"9.1"
#})

# Use on macos
#INCLUDE_PREFIX="local/include/"
#LIB_PREFIX="local/lib/"
#LIB_SUFFIX=".dylib"
#vtk_version="9.3"

# use on linux
#INCLUDE_PREFIX="include/"
#LIB_PREFIX="lib/x86_64-linux-gnu/"
#LIB_SUFFIX=".so"
#vtk_version="9.1"

def platform_glob(patterns):
  return select({
    "@platforms//os:linux":native.glob(patterns),
    "@platforms//os:macos":native.glob(["local/"+p for p in patterns])
  })

def platform_glob_lib(name_patterns):
  return select({
    "@platforms//os:linux":native.glob(["lib/x86_64-linux-gnu/lib"+p+".so" for p in name_patterns]),
    "@platforms//os:macos":native.glob(["local/lib/lib"+p+".dylib" for p in name_patterns])
  })

def platform_include_prefix(suffix=""):
  return select({
    "@platforms//os:linux":"include/"+suffix,
    "@platforms//os:macos":"local/include/"+suffix
  })

#def prefix_glob(include):
#  return select({
#    "@platforms//os:linux":glob(),
#    "@platforms//os:macos":glob()
#  })
