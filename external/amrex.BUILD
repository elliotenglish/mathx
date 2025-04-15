
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name="all",
    srcs=glob(["**"]),
    visibility=["//visibility:public"]
)

configure_make(
    name = "amrex",
    configure_options = ["--with-mpi=no",
                         "--enable-pic=yes"],
    lib_source=":all",
    configure_in_place=True,
    #copts=["-j16"],
    out_static_libs=["libamrex.a"],
    visibility=["//visibility:public"],
    linkopts=["-lgfortran"]
)

#./configure --with-mpi=no --enable-pic=yes --prefix=build
#make -j16
#make install
