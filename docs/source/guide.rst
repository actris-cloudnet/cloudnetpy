Developer's Guide
=================

CloudnetPy is hosted by Finnish Meteorological Institute (FMI) and
used to process cloud remote sensing data within ACTRIS research
infrastructure. We are happy to welcome the cloud remote sensing community
to provide improvements in the methods and their implementations, writing
tests and finding bugs.

How to contribute
-----------------

If you find a bug, or experience obviously incorrect behaviour, the right way
to report is to open an issue on `CloudnetPy's Github page <https://github.com/tukiains/cloudnetpy/issues>`_.
Describe the problem and show the steps to reproduce it. Suggest a solution if you can but
this is not necessary.

However, any suggestion that affects the *methodology*, and thus the outcome of the
processing, needs to be carefully reviewed by experts of the ACTRIS cloud
remote sensing community. The bigger the modification, the bigger the validation
work needed before the actual implementation can happen.

Coding guidelines
-----------------

- Use `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ standard.

- Check your code using, e.g., `Pylint <https://www.pylint.org/>`_.

- Write `Google-style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

- Follow `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_.

- Write *short* functions and classes.

- Use *meaningful* names for variables, functions, etc.

- Write *minimal* amount of comments. Your code should be self-explaining.

- Always unit-test your code!

Further reading:

- `Clean Code <https://www.oreilly.com/library/view/clean-code/9780136083238/>`_
- `Clean Code in Python <https://www.packtpub.com/eu/application-development/clean-code-python>`_
- `The Pragmatic Programmer <https://pragprog.com/book/tpp20/the-pragmatic-programmer-20th-anniversary-edition>`_





