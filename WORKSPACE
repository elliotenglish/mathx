workspace(name="code")

load("@bazel_tools//tools/build_defs/repo:git.bzl","new_git_repository")

new_local_repository(
    name = "sdl2",
    path = "/usr",
    build_file = "sdl2.BUILD"
)

new_local_repository(
    name = "silo",
    path = "/usr",
    build_file = "silo.BUILD"
)

new_local_repository(
    name = "opencv",
    path = "/usr",
    build_file = "opencv.BUILD"
)

new_local_repository(
    name="ceres-solver",
    #path="/usr/local/opt/ceres-solver",
    path="/usr",
    build_file="ceres-solver.BUILD"
)

new_local_repository(
    name="glog",
    #path="/usr/local/opt/glog",
    path="/usr",
    build_file="glog.BUILD"
)

new_local_repository(
    name="gflags",
    #path="/usr/local/opt/gflags",
    path="/usr",
    build_file="gflags.BUILD"
)

new_local_repository(
    name="eigen",
    path="/usr",
    build_file="eigen.BUILD"
)

new_local_repository(
    name="nanoflann",
    path="/usr",
    build_file="nanoflann.BUILD"
)

new_local_repository(
    name="rapidjson",
    path="/usr/local/opt/rapidjson",
    build_file="rapidjson.BUILD"
)

new_local_repository(
    name="nlohmann-json",
    path="/usr",
    build_file="nlohmann-json.BUILD"
)

new_local_repository(
    name="cxxopts",
    path="/usr",
    build_file="cxxopts.BUILD"
)

new_local_repository(
    name="vtk",
    path="/usr",
    build_file="vtk.BUILD"
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
    ],
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.2.0/rules_python-0.2.0.tar.gz",
    sha256 = "778197e26c5fbeb07ac2a2c5ae405b30f6cb7ad1f5510ea6fdac03bded96cc6f",
)

#####################################################
# The following 2 repositories and macro calls are needed to fix the following
# error when using newer versions of rules_foreign_cc:
#
# The repository '@@bazel_features_globals' could not be resolved: Repository
# '@@bazel_features_globals' is not defined.

http_archive(
    name = "bazel_features",
    sha256 = "95fb3cfd11466b4cad6565e3647a76f89886d875556a4b827c021525cb2482bb",
    strip_prefix = "bazel_features-1.10.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.10.0/bazel_features-v1.10.0.tar.gz",
)
load("@bazel_features//:deps.bzl", "bazel_features_deps")
bazel_features_deps()

http_archive(
    name = "rules_python",
    sha256 = "be04b635c7be4604be1ef20542e9870af3c49778ce841ee2d92fcb42f9d9516a",
    strip_prefix = "rules_python-0.35.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.35.0/rules_python-0.35.0.tar.gz",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

# http_archive(
#     name = "rules_foreign_cc",
#     # TODO: Get the latest sha256 value from a bazel debug message or the latest
#     #       release on the releases page: https://github.com/bazel-contrib/rules_foreign_cc/releases
#     #
#     #sha256 = "jlYF3C0WpCKcuPvjmFFLEFKFU+1PX3c3tmP92S9I4cI=",
#     strip_prefix = "rules_foreign_cc-0.11.1",
#     url = "https://github.com/bazel-contrib/rules_foreign_cc/archive/0.11.1.tar.gz",
# )

# local_repository(
#     name="rules_foreign_cc",
#     path="../rules_foreign_cc"
# )

new_git_repository(
    name="rules_foreign_cc",
    remote="git@github.com:elliotenglish/rules_foreign_cc.git",
    branch="fix_deps_environment_generation"
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies(register_built_tools=False)

#new_git_repository(
#    name="amrex",
#    build_file="amrex.BUILD",
#    remote="https://github.com/AMReX-Codes/amrex.git",
#    branch="development"
#)

# Use this to build from source
new_git_repository(
    name="libmesh",
    build_file="libmesh.BUILD",
    remote="https://github.com/libMesh/libmesh.git",
    #branch="devel",
    init_submodules=True,
    recursive_init_submodules=True,
    tag="v1.7.6"
)

# Use this to use a prebuild version
# new_local_repository(
#     name="libmesh",
#     build_file="libmesh.prebuilt.BUILD",
#     path="../libmesh"
# )

#TODO: This doesn't work because libmesh has recursive submodules
#http_archive(
#    name="libmesh",
#    build_file="libmesh.BUILD",
#    strip_prefix = "libmesh-1.7.6",
#    urls=["https://github.com/libMesh/libmesh/archive/refs/tags/v1.7.6.tar.gz"]
#)

http_archive(
    name = "com_google_googletest",
    # urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
    # strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls=["https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz"],
    strip_prefix="googletest-1.16.0"
)

http_archive(
    name = "autodiff",
    urls=["https://github.com/autodiff/autodiff/archive/refs/tags/v1.1.2.tar.gz"],
    strip_prefix="autodiff-1.1.2",
    repo_mapping={"@com_github_eigen_eigen":"@eigen"},
    build_file="autodiff.BUILD",
    integrity = "sha256-hvaKq9rh7tIUv78N2qGCx46hu5nk30BO+3uU0w4Gt0Q="
)
