import setuptools

def get_version():
    version = '1.1.1'
    return version

setuptools.setup(
    name="my-project",
    version=get_version(),
)
