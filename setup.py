# Copyright 2022 Intrinsic Innovation LLC.
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for PGMax package."""

import setuptools


def _get_version():
  with open('pgmax/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `pgmax/__init__.py`')


VERSION = _get_version()

if __name__ == '__main__':
  setuptools.setup(
      name='pgmax',
      version=VERSION,
      packages=setuptools.find_packages(),
      license='Apache 2.0',
      author='DeepMind',
      description=(
          'Loopy belief propagation for factor graphs on discrete variables'
          ' in JAX'
      ),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author_email='pgmax-dev@google.com',
      requires_python='>=3.7,<3.11',
  )
