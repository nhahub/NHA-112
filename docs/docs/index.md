# dataEngineerProject documentation!

## Description

A short description of the project.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `Mycontainer/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `Mycontainer/data/` to `data/`.


