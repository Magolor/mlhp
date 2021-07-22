chmod -R 777 ./
rm -rf pkg
mkdir pkg
python setup.py bdist_wheel
mv dist/* pkg/
rm -rf __pycache__
rm -rf build
rm -rf dist
pip install --user pkg/mlhp_magolor-0.1.0.0-py3-none-any.whl --force-reinstall