import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name='ugaf',
	version='0.0.1',
	author='Ashkan Dehghan',
	author_email='ash.dehghan@gmail.com',
	description='Unsupervised Graph Analysis Framework.',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url='https://github.com/elmspace/ugaf/tree/master',
	license='BSD-2',
	packages=['ugaf'],
	install_requires=["pandas==2.0.3", "arrow==1.2.3", "tqdm==4.65.0"]
)