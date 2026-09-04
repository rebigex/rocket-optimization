# Third-party components

## openMotor — GPLv3, not redistributed here

The internal ballistics engine is [openMotor](https://github.com/reilleya/openMotor)
by Andrew Reilley, licensed GPLv3. This repository **does not contain a copy of it**.
`scripts/setup_env.sh` clones it into `vendor/openMotor` at the pinned commit
`0dfb3f1dd4f843499c7f71dc85a3dfde5dd15c6a` and builds its Cython extension in place.



## Plotly.js — MIT, not redistributed here

The web app serves Plotly from disk so it works offline. `setup_env.sh` copies
`plotly.min.js` out of the installed `plotly` pip package into
`app/static/vendor/`; it is not committed.
