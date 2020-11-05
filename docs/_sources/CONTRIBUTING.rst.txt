Guideline of contribution
=========================

| Thank you always for your comments and using this repository!
| We hope this project contribute to COVID-19 outbreak research and we
  will overcome the outbreak in the near future.

When you contribute to covid-sir repository including CovsirPhy package,
please kindly follow CODE\_OF\_CONDUCT.md and the steps explained in
this document. If you have any questions or any request to change about
this guideline, please inform the author via issue page.

Terms: - 'Maintainers' maintain this repository and merge the pull
requests, update `sphinx
document <https://lisphilar.github.io/covid19-sir/>`__ and version
number. - 'Developers' create pull requests. - 'Users' uses this package
(and create issues).

Currently, 'maintainer(s)' is/are the author (@lisphilar) in this
document.

Issues
------

All users can `create an
issue <https://github.com/lisphilar/covid19-sir/issues>`__ when you want
to - ask a question - request fix a bug, - request update of
documentation, - request a new feature, - request a new data loader or
new dataset, - communicate with the maintainers and the user community.

Please ensure that the issue - has one question/request, - uses
appropriate issue template if available, - has appropriate title, and -
explains your execution environment and codes.

When requesting a new data loader, please confirm that the dataset
follows ALCOA (Attributable, Legible, Contemporaneous, Original,
Accurate) and we can retrieve the data with permitted APIs. If the data
loaders included in this repository use un-permitted APIs, please inform
maintainers. We will fix it as soon as possible.

If the issue has already been discussed, maintainers will mark it with
'duplicate' label.

Pull request
------------

If you have any ideas, please `create a pull
request <https://github.com/lisphilar/covid19-sir/pulls>`__ after
explaining your idea in an issue. Please ensure that - you are using the
latest version of CovsirPhy, - the codes are readable, - (if needed)
test codes are added, - tests were successfully completed, and - the
revision is well documented in docstring/README and so on.

Test command is explained in `Installation and dataset
preparation <https://lisphilar.github.io/covid19-sir/INSTALLATION.html>`__
for developers.

Please use appropriate template if available, and specify the related
issues and the change.

We use GitHub flow here. Mater (or main) branch can be deployed always
as the latest development version. Name of branch for pull requests
should be linked to issue numbers, like "issue100".

Update of version number and sphinx documents is not necessary for
developers. After reviewing by assigned reviewers, maintainers will
update them when merge the pull request.

Versioning
----------

CovsirPhy follows `Semantic Versioning 2.0.0 <https://semver.org/>`__:

Milestones of minor update (from X.0.Z to X.1.Z) are documented in
`milestones of
issues <https://github.com/lisphilar/covid19-sir/milestones>`__. When
the revisions do not change the codes of CovsirPhy, version number will
not be updated.

Maintainers will test the codes, update `sphinx
document <https://lisphilar.github.io/covid19-sir/>`__, update the
version number, upload to `PyPI: The Python Package
Index <https://pypi.org/>`__, and create `a release
note <https://github.com/lisphilar/covid19-sir/releases>`__.
