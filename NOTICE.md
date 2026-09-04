# Third-party components

## openMotor — GPLv3, not redistributed here

The internal ballistics engine is [openMotor](https://github.com/reilleya/openMotor)
by Andrew Reilley, licensed GPLv3. This repository **does not contain a copy of it**.
`scripts/setup_env.sh` clones it into `vendor/openMotor` at the pinned commit
`0dfb3f1dd4f843499c7f71dc85a3dfde5dd15c6a` and builds its Cython extension in place.

That is deliberate. Shipping openMotor inside this repository would make the combined
work a distribution of GPLv3 code, and the licence of everything here would have to
follow. Cloning it at setup keeps the two separable.

**Worth knowing before making this repository public:** the code here imports
`motorlib` directly. Whether that makes it a derivative work under the GPL is the
long-running question about linking, and reasonable people disagree. If you intend to
publish and want to be conservative, licensing this project GPLv3 too removes the
question entirely.

## Plotly.js — MIT, not redistributed here

The web app serves Plotly from disk so it works offline. `setup_env.sh` copies
`plotly.min.js` out of the installed `plotly` pip package into
`app/static/vendor/`; it is not committed.
