
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")
#load("@rules_cc//cc:defs.bzl", "ACTION_NAMES")

#features = [
#    ACTION_NAMES.c_compile,
#    ACTION_NAMES.cpp_compile
#]

filegroup(
    name="all",
    srcs=glob(["**"]),
    visibility=["//visibility:public"]
)

common_options=[
    "CFLAGS=","CXXFLAGS=", #Hack to fix redacted defines
    "AR=", #Hack to fix libtool error on macos
    #"--disable-eigen", #Subsequent dependencies will complain that eigen can't be found
    "--with-eigen-include=$$EXT_BUILD_DEPS/include",
    #"--disable-vtk", #Subsequent dependencies will complain that vtk can't be found
    "--with-vtk-include=$$EXT_BUILD_DEPS/include",
    "--with-vtk-lib=$$EXT_BUILD_DEPS/lib",
    #"--verbose"
]

configure_make(
    name = "libmesh",
    env = {
        "HOME":"" #Hack to fix HOME not being set and accessible to MPI Opal get home directory
    },
    configure_options = select({
        "@platforms//os:linux":common_options,
        "@platforms//os:macos":common_options+["--disable-mpi"]
    }),
    lib_source=":all",
    configure_in_place=True,
    args=["-j4"],#["-j"],
    out_shared_libs=select({
        "@platforms//os:linux":["libmesh_opt.so.0","libtimpi_opt.so.5"],
        "@platforms//os:macos":["libmesh_opt.0.dylib","libtimpi_opt.5.dylib"]
    }),
    deps=["@eigen//:eigen","@vtk//:vtk"],
    visibility=["//visibility:public"]
)
