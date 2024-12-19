import os
import subprocess


def build(working_path: str):
    # Chage to working path.
    os.chdir(working_path)
    print("working in dir: ", working_path)

    # Check if build directory exists, if exists, remove it.
    build_dir = os.path.join(working_path, "build")
    if os.path.exists(build_dir):
        print("build dir {} exists, remove it.".format(build_dir))
        os.rmdir(build_dir)

    # Build build path and move into it.
    os.mkdir(build_dir)
    print("make build dir: ", build_dir)

    os.chdir(build_dir)
    print("working in dir: ", build_dir)

    # Make config.
    res = subprocess.run(
        [
            "cmake",
            "-DCMAKE_INSTALL_PREFIX=/usr",
            "..",
            " -DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_SKIP_RPATH=TRUE",
        ],
        capture_output=True,
    )
    if res.returncode != 0:
        print("cmake configurate failed. ", res.stderr.decode())
        return res.returncode

    # Make
    res = subprocess.run(["make"], capture_output=True)
    if res.returncode != 0:
        print("make failed. ", res.stderr.decode())
        return res.returncode

    # Install
    res = subprocess.run(["make", "install"], capture_output=True)
    if res.returncode != 0:
        print("make install failed. ", res.stderr.decode())
        return res.returncode

    return 0


if __name__ == "__main__":
    import sys

    packages = ["Common", "DVPPLite", "Media", "OMExecute"]
    for p in packages:
        package_dir = os.path.join(os.getcwd(), p)
        ret = build(package_dir)
        if ret != 0:
            sys.exit(ret)

    sys.exit(0)
