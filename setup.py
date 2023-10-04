import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name='ugaf',
	version='0.0.3',
	author='Ashkan Dehghan',
	author_email='ash.dehghan@gmail.com',
	description='Unsupervised Graph Analysis Framework.',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url='https://github.com/elmspace/ugaf/tree/master',
	license='BSD-2',
	packages=['ugaf'],
	install_requires=["pandas==2.0.3",
	"arrow==1.2.3",
	"tqdm==4.65.0",
	"scikit-learn==1.3.0",
	"matplotlib==3.7.2",
	"scipy==1.11.2",
	"tensorflow==2.13.0",
	"networkx==3.1",
	"loguru==0.7.2",
	"node2vec==0.4.6"]
)