/* Panels and the profiles that group them.
 *
 * A profile is just a list of panel ids, so adding one is a one-line change.
 * Colours come from the same validated palette the printed report uses, and
 * scatter plots stay within its first three slots -- past three, adjacent hues
 * stop being reliably separable for colourblind readers. */

const Charts = (() => {

  const SERIES = ['#2a78d6', '#eb6834', '#1baf7a'];
  const LIMIT  = '#e34948';
  const RAMP   = ['#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];

  function css(name) {
    return getComputedStyle(document.body).getPropertyValue(name).trim();
  }

  function theme() {
    return {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'system-ui, -apple-system, sans-serif', size: 11,
              color: css('--ink-2') },
      margin: { l: 54, r: 16, t: 10, b: 42 },
      xaxis: { gridcolor: css('--line-2'), zerolinecolor: css('--line'),
               linecolor: css('--line'), automargin: true },
      yaxis: { gridcolor: css('--line-2'), zerolinecolor: css('--line'),
               linecolor: css('--line'), automargin: true },
      legend: { orientation: 'h', y: -0.22, font: { size: 10.5 } },
      hoverlabel: { bgcolor: css('--surface'), bordercolor: css('--line'),
                    font: { size: 11, color: css('--ink') } },
      showlegend: false
    };
  }

  const CONFIG = { displayModeBar: false, responsive: true };

  function draw(node, traces, extra) {
    const layout = Object.assign(theme(), extra || {});
    Plotly.react(node, traces, layout, CONFIG);
  }

  /* ------------------------------------------------------------ helpers */

  function metricLabel(key) {
    const m = (App.state.metrics || {})[key];
    return m ? m.label : key;
  }

  function metricValue(row, key) {
    // Display units differ from storage units for pressure and mass flux.
    if (key === 'max_pressure' || key === 'avg_pressure') {
      return row[key] / 6894.757293168361;
    }
    if (key === 'peak_mass_flux') return row[key] / 703.0696;
    return row[key];
  }

  function metricUnit(key) {
    const m = (App.state.metrics || {})[key];
    return m && m.unit ? m.unit : '';
  }

  /* ------------------------------------------------------------- panels */

  const PANELS = {

    thrustCurve: {
      title: 'Thrust curve',
      sub: 'Selected design, simulated at full fidelity',
      render(node, ctx) {
        const d = ctx.design, b = ctx.baseline;
        const traces = [];
        if (b && b.curves) traces.push({
          x: b.curves.time, y: b.curves.thrust, mode: 'lines', name: 'your motor',
          line: { color: css('--ink-3'), width: 1.5, dash: 'dot' }
        });
        if (d && d.curves) traces.push({
          x: d.curves.time, y: d.curves.thrust, mode: 'lines', name: 'optimized',
          line: { color: SERIES[0], width: 2.2 }
        });
        draw(node, traces, {
          showlegend: true,
          xaxis: Object.assign(theme().xaxis, { title: 'Time (s)' }),
          yaxis: Object.assign(theme().yaxis, { title: 'Thrust (N)', rangemode: 'tozero' })
        });
      }
    },

    pressureKn: {
      title: 'Chamber pressure and Kn',
      sub: 'Two stacked panels — never two scales on one axis',
      render(node, ctx) {
        const d = ctx.design; if (!d || !d.curves) return;
        const psi = d.curves.pressure.map(p => p / 6894.757293168361);
        const t = theme();
        const traces = [
          { x: d.curves.time, y: psi, mode: 'lines', name: 'pressure',
            line: { color: SERIES[1], width: 2 }, xaxis: 'x', yaxis: 'y' },
          { x: d.curves.time, y: d.curves.kn, mode: 'lines', name: 'Kn',
            line: { color: SERIES[2], width: 2 }, xaxis: 'x2', yaxis: 'y2' }
        ];
        const limits = [];
        (ctx.constraints || []).forEach(c => {
          if (c.metric === 'max_pressure')
            limits.push(hline(c.value / 6894.757293168361, 'y'));
          if (c.metric === 'peak_kn') limits.push(hline(c.value, 'y2'));
        });
        draw(node, traces, {
          grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
          margin: { l: 54, r: 16, t: 8, b: 38 },
          xaxis:  Object.assign({}, t.xaxis, { anchor: 'y', showticklabels: false }),
          yaxis:  Object.assign({}, t.yaxis, { title: 'psi', domain: [0.56, 1] }),
          xaxis2: Object.assign({}, t.xaxis, { anchor: 'y2', title: 'Time (s)' }),
          yaxis2: Object.assign({}, t.yaxis, { title: 'Kn', domain: [0, 0.44] }),
          shapes: limits
        });
      }
    },

    crossSection: {
      title: 'Motor cross-section',
      sub: 'To scale, forward at left',
      render(node, ctx) { node.innerHTML = crossSectionSVG(ctx.design, ctx.baseline); }
    },

    specSheet: {
      title: 'Specification',
      sub: 'Optimized design against your current motor',
      render(node, ctx) { node.innerHTML = deltaTable(ctx.design, ctx.baseline); }
    },

    marginBars: {
      title: 'How close it runs to each limit',
      sub: 'Full bar means the limit is reached exactly',
      render(node, ctx) {
        const d = ctx.design; if (!d) return;
        node.innerHTML = (ctx.constraints || []).filter(c => c.enabled)
          .map(c => marginRow(c, d)).join('') ||
          '<p class="sub">No limits set.</p>';
      }
    },

    /* ------------------------------------------------- trade-off explorer */

    paretoFront: {
      title: 'Your options',
      sub: 'Click any point to load that motor into Design Review',
      render(node, ctx) {
        const designs = ctx.designs || [];
        if (!designs.length) { node.innerHTML = '<p class="sub">No feasible designs.</p>'; return; }
        const [ax, ay] = ctx.axes;
        const traces = [{
          x: designs.map(d => metricValue(d, ax)),
          y: designs.map(d => metricValue(d, ay)),
          mode: 'lines+markers', type: 'scatter',
          marker: { size: 9, color: SERIES[0], line: { color: css('--surface'), width: 1.5 } },
          line: { color: SERIES[0], width: 1.5 },
          text: designs.map(d => d.designation), name: 'options',
          hovertemplate: '%{text}<br>' + metricLabel(ax) + ': %{x:,.0f}<br>' +
                         metricLabel(ay) + ': %{y:,.0f}<extra></extra>'
        }];
        if (ctx.baseline) traces.push({
          x: [metricValue(ctx.baseline, ax)], y: [metricValue(ctx.baseline, ay)],
          mode: 'markers', name: 'your motor',
          marker: { size: 13, symbol: 'diamond', color: LIMIT,
                    line: { color: css('--surface'), width: 1.5 } },
          hovertemplate: 'your motor<extra></extra>'
        });
        draw(node, traces, {
          showlegend: true,
          xaxis: Object.assign(theme().xaxis, { title: axisTitle(ax) }),
          yaxis: Object.assign(theme().yaxis, { title: axisTitle(ay) })
        });
        node.on('plotly_click', ev => {
          const p = ev.points[0];
          if (p.curveNumber === 0) App.selectDesign(p.pointIndex);
        });
      }
    },

    populationCloud: {
      title: 'Every design tried',
      sub: 'Grey failed a limit; blue met them all',
      render(node, ctx) {
        const pop = ctx.population || [];
        if (!pop.length) { node.innerHTML = '<p class="sub">No population recorded.</p>'; return; }
        const [ax, ay] = ctx.axes;
        const ok = pop.filter(r => r.feasible), bad = pop.filter(r => !r.feasible);
        const traces = [
          { x: bad.map(r => metricValue(r, ax)), y: bad.map(r => metricValue(r, ay)),
            mode: 'markers', name: 'over a limit',
            marker: { size: 3.5, color: css('--line'), opacity: .85 }, hoverinfo: 'skip' },
          { x: ok.map(r => metricValue(r, ax)), y: ok.map(r => metricValue(r, ay)),
            mode: 'markers', name: 'legal',
            marker: { size: 4, color: SERIES[0], opacity: .55 }, hoverinfo: 'skip' }
        ];
        if (ctx.baseline) traces.push({
          x: [metricValue(ctx.baseline, ax)], y: [metricValue(ctx.baseline, ay)],
          mode: 'markers', name: 'your motor',
          marker: { size: 13, symbol: 'diamond', color: LIMIT,
                    line: { color: css('--surface'), width: 1.5 } }
        });
        draw(node, traces, {
          showlegend: true,
          xaxis: Object.assign(theme().xaxis, { title: axisTitle(ax) }),
          yaxis: Object.assign(theme().yaxis, { title: axisTitle(ay) })
        });
      }
    },

    parallelCoords: {
      title: 'What the good designs have in common',
      sub: 'Each line is one legal design, coloured by how well it scored',
      render(node, ctx) {
        const pop = (ctx.population || []).filter(r => r.feasible);
        if (pop.length < 5) {
          node.innerHTML = '<p class="sub">Not enough legal designs to compare yet.</p>';
          return;
        }
        const vars = (ctx.searched || []).filter(v => pop[0][v] !== undefined);
        if (!vars.length) { node.innerHTML = '<p class="sub">No searched dimensions.</p>'; return; }
        node.innerHTML = parallelSVG(pop, vars, ctx.axes[0]);
      }
    },

    objectiveSpread: {
      title: 'Spread of results',
      sub: 'Where the search spent its time',
      render(node, ctx) {
        const pop = ctx.population || [];
        if (!pop.length) { node.innerHTML = '<p class="sub">No population recorded.</p>'; return; }
        const key = ctx.axes[0];
        const ok = pop.filter(r => r.feasible).map(r => metricValue(r, key));
        const bad = pop.filter(r => !r.feasible).map(r => metricValue(r, key));
        draw(node, [
          { x: bad, type: 'histogram', name: 'over a limit',
            marker: { color: css('--line') }, opacity: .9, nbinsx: 40 },
          { x: ok, type: 'histogram', name: 'legal',
            marker: { color: SERIES[0] }, opacity: .85, nbinsx: 40 }
        ], {
          barmode: 'overlay', showlegend: true,
          xaxis: Object.assign(theme().xaxis, { title: axisTitle(key) }),
          yaxis: Object.assign(theme().yaxis, { title: 'designs' })
        });
      }
    },

    /* -------------------------------------------------------- diagnostics */

    convergence: {
      title: 'Search progress',
      sub: 'Best legal score found, against simulations spent',
      render(node, ctx) {
        const c = ctx.convergence || [];
        if (!c.length) { node.innerHTML = '<p class="sub">No convergence data.</p>'; return; }
        draw(node, [{
          x: c.map(p => p.n), y: c.map(p => p.best), mode: 'lines',
          line: { color: SERIES[0], width: 2 }, name: 'best so far'
        }], {
          xaxis: Object.assign(theme().xaxis, { title: 'simulations', type: 'log' }),
          yaxis: Object.assign(theme().yaxis, { title: 'score' })
        });
      }
    },

    parity: {
      title: 'Model accuracy',
      sub: 'Predicted against simulated, on designs held back from training',
      render(node, ctx) {
        const s = ctx.surrogate;
        if (!s || !s.parity) {
          node.innerHTML = '<p class="sub">Only trained in full trade-off mode. ' +
            'Tick “Map the full trade-off” to see this.</p>'; return;
        }
        const key = ctx.axes[0] in s.parity ? ctx.axes[0] : Object.keys(s.parity)[0];
        const p = s.parity[key];
        const lo = Math.min(...p.actual), hi = Math.max(...p.actual);
        draw(node, [
          { x: p.actual, y: p.predicted, mode: 'markers', name: key,
            marker: { size: 4, color: SERIES[0], opacity: .35 }, hoverinfo: 'skip' },
          { x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'perfect',
            line: { color: css('--ink-3'), width: 1, dash: 'dash' }, hoverinfo: 'skip' }
        ], {
          xaxis: Object.assign(theme().xaxis, { title: 'simulated ' + metricLabel(key) }),
          yaxis: Object.assign(theme().yaxis, { title: 'predicted' })
        });
      }
    },

    importance: {
      title: 'Which dimension matters most',
      sub: 'Drop in model accuracy when that value is shuffled',
      render(node, ctx) {
        const s = ctx.surrogate;
        if (!s || !s.importances) {
          node.innerHTML = '<p class="sub">Only measured in full trade-off mode.</p>'; return;
        }
        const rows = s.importances.slice(0, 10).reverse();
        draw(node, [{
          type: 'bar', orientation: 'h',
          y: rows.map(r => shortVar(r.feature)), x: rows.map(r => r.importance),
          marker: { color: SERIES[0] }, hovertemplate: '%{y}: %{x:.3f}<extra></extra>'
        }], {
          margin: { l: 108, r: 16, t: 8, b: 38 },
          xaxis: Object.assign(theme().xaxis, { title: 'importance' }),
          yaxis: Object.assign(theme().yaxis, { automargin: true })
        });
      }
    },

    constraintActivity: {
      title: 'Which limit is holding you back',
      sub: 'Share of legal designs sitting within 2% of each limit',
      render(node, ctx) {
        const rows = ctx.constraintActivity || [];
        if (!rows.length) { node.innerHTML = '<p class="sub">No limits set.</p>'; return; }
        const sorted = rows.slice().sort((a, b) => a.binding_fraction - b.binding_fraction);
        draw(node, [{
          type: 'bar', orientation: 'h',
          y: sorted.map(r => r.label), x: sorted.map(r => r.binding_fraction * 100),
          marker: { color: sorted.map(r => r.binding_fraction > 0.4 ? LIMIT : SERIES[0]) },
          hovertemplate: '%{y}: %{x:.0f}% of legal designs are up against it<extra></extra>'
        }], {
          margin: { l: 130, r: 16, t: 8, b: 38 },
          xaxis: Object.assign(theme().xaxis, { title: '% of legal designs at the limit' }),
          yaxis: Object.assign(theme().yaxis, { automargin: true })
        });
      }
    },

    /* ----------------------------------------------------- compare & safety */

    compareThrust: {
      title: 'Before and after',
      sub: 'Your motor against the selected design',
      render(node, ctx) { PANELS.thrustCurve.render(node, ctx); }
    },

    grainFlux: {
      title: 'Mass flux in each grain',
      sub: 'The aft grain always runs hottest — that is the one the limit is about',
      render(node, ctx) {
        const d = ctx.design;
        if (!d || !d.curves || !d.curves.mass_flux || !d.curves.mass_flux.length) {
          node.innerHTML = '<p class="sub">No flux data.</p>'; return;
        }
        const n = d.curves.mass_flux.length;
        const traces = d.curves.mass_flux.map((series, i) => ({
          x: d.curves.time, y: series.map(v => v / 703.0696), mode: 'lines',
          name: 'grain ' + (i + 1),
          line: { color: RAMP[Math.round(i * (RAMP.length - 1) / Math.max(n - 1, 1))], width: 1.8 }
        }));
        const limits = (ctx.constraints || [])
          .filter(c => c.metric === 'peak_mass_flux' && c.enabled)
          .map(c => hline(c.value / 703.0696, 'y'));
        draw(node, traces, {
          showlegend: true, shapes: limits,
          xaxis: Object.assign(theme().xaxis, { title: 'Time (s)' }),
          yaxis: Object.assign(theme().yaxis, { title: 'lb/in²·s', rangemode: 'tozero' })
        });
      }
    },

    tornado: {
      title: 'What moves the needle',
      sub: 'Change in the leading goal from one step either way',
      render(node, ctx) {
        const rows = (ctx.sensitivity || []).slice(0, 9).reverse();
        if (!rows.length) { node.innerHTML = '<p class="sub">No sensitivity data.</p>'; return; }
        draw(node, [
          { type: 'bar', orientation: 'h', name: 'one step smaller',
            y: rows.map(r => shortVar(r.variable)), x: rows.map(r => r.down),
            marker: { color: SERIES[1] },
            hovertemplate: '%{y} smaller: %{x:+,.1f}<extra></extra>' },
          { type: 'bar', orientation: 'h', name: 'one step larger',
            y: rows.map(r => shortVar(r.variable)), x: rows.map(r => r.up),
            marker: { color: SERIES[0] },
            hovertemplate: '%{y} larger: %{x:+,.1f}<extra></extra>' }
        ], {
          barmode: 'overlay', showlegend: true,
          margin: { l: 108, r: 16, t: 8, b: 38 },
          xaxis: Object.assign(theme().xaxis, { title: 'change in ' + metricLabel(ctx.axes[0]),
                                                zeroline: true }),
          yaxis: Object.assign(theme().yaxis, { automargin: true })
        });
      }
    },

    robustness: {
      title: 'What happens when you build it',
      sub: 'The same design made many times, with your tolerances applied',
      render(node, ctx) {
        const r = ctx.robustness;
        if (!r) {
          node.innerHTML = `<div class="robust">
            <p class="lead">The optimizer works from nominal dimensions, so every
            design it returns sits exactly on whatever limits you set. This simulates
            the design as it would actually come out of the shop — core diameters,
            throat, and propellant batch each varying by the tolerances in the rail —
            and reports how often it still stays legal.</p>
            <div><button type="button" class="chip" id="btnRobust">Check robustness</button></div>
          </div>`;
          const b = node.querySelector('#btnRobust');
          if (b) b.addEventListener('click', () => ctx.onCheckRobustness(node));
          return;
        }
        if (!r.available) {
          node.innerHTML = `<p class="sub">${r.reason || 'No robustness data.'}</p>`;
          return;
        }
        const pct = 100 * r.pass_rate;
        const cls = pct >= 90 ? 'ok' : (pct >= 70 ? '' : 'bad');
        const bars = (r.per_limit || []).map(l => {
          const p = 100 * l.exceed_probability;
          const fill = p < 1 ? 'none' : (p < 10 ? 'low' : '');
          return `<div class="bar-row">
            <span class="label">${l.label}</span>
            <span class="track"><span class="fill ${fill}"
              style="width:${Math.min(p, 100).toFixed(1)}%"></span></span>
            <span class="pct">${p.toFixed(0)}% over</span></div>`;
        }).join('');
        node.innerHTML = `<div class="robust">
          <div class="headline">
            <span class="rate ${cls}">${pct.toFixed(0)}%</span>
            <span class="rate-note">of ${r.samples} builds stay inside every limit<br>
              95% confidence ${(100 * r.pass_low).toFixed(0)}–${(100 * r.pass_high).toFixed(0)}%</span>
          </div>
          <div>${bars}</div>
          <div><button type="button" class="chip" id="btnRobust">Run again</button></div>
        </div>`;
        const b = node.querySelector('#btnRobust');
        if (b) b.addEventListener('click', () => ctx.onCheckRobustness(node));
      }
    },

    robustnessSpread: {
      title: 'Where the builds land',
      sub: 'The limit that goes over most often, across every simulated build',
      render(node, ctx) {
        const r = ctx.robustness;
        if (!r || !r.available || !r.per_limit || !r.per_limit.length) {
          node.innerHTML = '<p class="sub">Run the robustness check to see this.</p>';
          return;
        }
        const worst = r.per_limit[0];
        // Samples arrive in SI; metricValue converts one row, so wrap each in
        // the shape it expects rather than repeating the unit table here.
        const toShown = v => metricValue({ [worst.metric]: v }, worst.metric);
        const shown = (worst.samples || []).map(toShown);
        const limit = toShown(worst.limit);
        const over = shown.filter(v => worst.op === '<=' ? v > limit : v < limit);
        const under = shown.filter(v => worst.op === '<=' ? v <= limit : v >= limit);
        draw(node, [
          { x: under, type: 'histogram', name: 'legal',
            marker: { color: SERIES[0] }, opacity: .85, nbinsx: 44 },
          { x: over, type: 'histogram', name: 'over the limit',
            marker: { color: LIMIT }, opacity: .85, nbinsx: 44 }
        ], {
          barmode: 'overlay', showlegend: true,
          shapes: [hline(limit, 'y')].map(sh => Object.assign(sh, {
            xref: 'x', yref: 'paper', x0: limit, x1: limit, y0: 0, y1: 1,
            line: { color: LIMIT, width: 1.6, dash: 'dash' } })),
          xaxis: Object.assign(theme().xaxis, { title: axisTitle(worst.metric) }),
          yaxis: Object.assign(theme().yaxis, { title: 'builds' })
        });
      }
    },

    optionsTable: {
      title: 'All options found',
      sub: 'Click a row to inspect it; export writes a .ric you can open in openMotor',
      render(node, ctx) { node.innerHTML = optionsTable(ctx); App.wireOptionsTable(node); }
    }
  };

  /* --------------------------------------------------------- the profiles */

  const PROFILES = [
    { id: 'design',      label: 'Design Review',
      panels: [['thrustCurve', 1], ['pressureKn', 1], ['crossSection', 2],
               ['specSheet', 1], ['marginBars', 1]] },
    { id: 'tradeoff',    label: 'Trade-off Explorer',
      panels: [['paretoFront', 1], ['populationCloud', 1], ['parallelCoords', 2],
               ['objectiveSpread', 1], ['optionsTable', 1]] },
    { id: 'diagnostics', label: 'Optimizer Diagnostics',
      panels: [['convergence', 1], ['constraintActivity', 1], ['parity', 1],
               ['importance', 1]] },
    { id: 'compare',     label: 'Compare & Safety',
      panels: [['compareThrust', 2], ['specSheet', 1], ['grainFlux', 1],
               ['robustness', 1], ['robustnessSpread', 1], ['tornado', 2]] }
  ];

  /* ------------------------------------------------------------ fragments */

  function hline(y, axisRef) {
    return { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: axisRef, y0: y, y1: y,
             line: { color: LIMIT, width: 1, dash: 'dot' } };
  }

  function axisTitle(key) {
    const u = metricUnit(key);
    return metricLabel(key) + (u ? ' (' + u + ')' : '');
  }

  function shortVar(name) {
    return name.replace('core_', 'core ').replace('exit_frac', 'exit')
               .replace('_', ' ');
  }

  function parallelSVG(rows, vars, colourBy) {
    const W = 760, H = 312, padL = 26, padR = 26, padT = 48, padB = 46;
    const n = vars.length;
    const step = (W - padL - padR) / Math.max(n - 1, 1);
    const y0 = padT, y1 = H - padB;

    // Each axis gets its own scale; a shared one would flatten every
    // dimension whose range is small next to the largest.
    const scales = vars.map(v => {
      const values = rows.map(r => r[v]);
      let lo = Math.min(...values), hi = Math.max(...values);
      if (hi - lo < 1e-12) { hi = lo + 1; }
      return { lo, hi };
    });
    const cValues = rows.map(r => metricValue(r, colourBy));
    const cLo = Math.min(...cValues), cHi = Math.max(...cValues);
    const colourAt = v => {
      const t = (cHi - cLo) < 1e-12 ? 1 : (v - cLo) / (cHi - cLo);
      return RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
    };
    const shown = rows.length > 700 ? rows.filter((_, i) => i % Math.ceil(rows.length / 700) === 0) : rows;

    let lines = '';
    shown.forEach(r => {
      const pts = vars.map((v, i) => {
        const s = scales[i];
        const y = y1 - ((r[v] - s.lo) / (s.hi - s.lo)) * (y1 - y0);
        return (padL + i * step).toFixed(1) + ',' + y.toFixed(1);
      }).join(' ');
      lines += `<polyline points="${pts}" fill="none" stroke="${colourAt(metricValue(r, colourBy))}"
                stroke-width="0.9" opacity="0.42"/>`;
    });

    let axes = '';
    vars.forEach((v, i) => {
      const x = (padL + i * step).toFixed(1);
      const s = scales[i];
      const fmt = q => (v === 'exit_frac' ? q.toFixed(2)
                        : (q / App.unitScale()).toFixed(App.state.unit === 'in' ? 2 : 1));
      axes += `<line x1="${x}" y1="${y0}" x2="${x}" y2="${y1}" stroke="var(--line)" stroke-width="1"/>
        <text x="${x}" y="${y0 - 9}" font-size="8.5" fill="var(--ink-3)" text-anchor="middle"
          font-family="ui-monospace, monospace">${fmt(s.hi)}</text>
        <text x="${x}" y="${y1 + 13}" font-size="8.5" fill="var(--ink-3)" text-anchor="middle"
          font-family="ui-monospace, monospace">${fmt(s.lo)}</text>
        <text x="${x}" y="${y1 + 30}" font-size="9.5" fill="var(--ink-2)" text-anchor="middle"
          font-family="system-ui, sans-serif">${shortVar(v)}</text>`;
    });

    const keyX = padL + 34;
    const swatches = RAMP.map((c, i) =>
      `<rect x="${(keyX + i * 14).toFixed(0)}" y="10" width="13" height="6" fill="${c}"/>`).join('');
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
      aria-label="Parallel coordinates of legal designs">${lines}${axes}${swatches}
      <text x="${keyX - 5}" y="16" font-size="8.5" fill="var(--ink-3)" text-anchor="end"
        font-family="system-ui, sans-serif">worse</text>
      <text x="${keyX + RAMP.length * 14 + 5}" y="16" font-size="8.5" fill="var(--ink-3)"
        font-family="system-ui, sans-serif">better</text></svg>`;
  }

  function crossSectionSVG(design, baseline) {
    const d = design || baseline;
    if (!d || !d.cores) return '<p class="sub">No geometry.</p>';
    const lengths = d.grain_lengths || d.cores.map(() => 0.1524);
    const bore = d.grain_diameter || 0.0822;
    const total = lengths.reduce((a, b) => a + b, 0) + d.throat * 4;
    const W = 640, H = 190, pad = 14;
    const sx = (W - 2 * pad) / total, sy = (H - 2 * pad) / bore;
    const s = Math.min(sx, sy);
    const cy = H / 2;
    let x = pad, out = '';
    d.cores.forEach((core, i) => {
      const L = lengths[i] * s, R = bore * s / 2, r = core * s / 2;
      out += `<rect x="${x.toFixed(1)}" y="${(cy - R).toFixed(1)}" width="${L.toFixed(1)}"
              height="${(R - r).toFixed(1)}" fill="var(--accent)" opacity="0.82"/>`;
      out += `<rect x="${x.toFixed(1)}" y="${(cy + r).toFixed(1)}" width="${L.toFixed(1)}"
              height="${(R - r).toFixed(1)}" fill="var(--accent)" opacity="0.82"/>`;
      out += `<line x1="${x.toFixed(1)}" y1="${(cy - R).toFixed(1)}" x2="${x.toFixed(1)}"
              y2="${(cy + R).toFixed(1)}" stroke="var(--surface)" stroke-width="1.5"/>`;
      out += `<text x="${(x + L / 2).toFixed(1)}" y="${(cy + R + 12).toFixed(1)}"
              font-size="8.5" fill="var(--ink-3)" text-anchor="middle"
              font-family="ui-monospace, monospace">${App.fmtLen(core)}</text>`;
      x += L;
    });
    // nozzle: convergent cone into the throat, then the exit cone
    const R = bore * s / 2, rt = d.throat * s / 2, re = d.exit * s / 2;
    const conv = d.throat * 1.6 * s, div = d.throat * 2.6 * s;
    out += `<polygon points="${x},${cy - R} ${x + conv},${cy - rt} ${x + conv + div},${cy - re}
            ${x + conv + div},${cy + re} ${x + conv},${cy + rt} ${x},${cy + R}"
            fill="var(--ink-3)" opacity="0.55"/>`;
    out += `<line x1="${pad}" y1="${cy}" x2="${(x + conv + div).toFixed(1)}" y2="${cy}"
            stroke="var(--ink-3)" stroke-dasharray="3 3" stroke-width="0.8" opacity="0.6"/>`;
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
      aria-label="Motor cross-section, forward at left">${out}</svg>
      <p class="sub" style="margin-top:6px">Throat ${App.fmtLen(d.throat)} ·
      exit ${App.fmtLen(d.exit)} · expansion ${(d.exit / d.throat * d.exit / d.throat).toFixed(2)}</p>`;
  }

  const SPEC_ROWS = [
    ['initial_thrust', 'Initial thrust', 'N', 0],
    ['total_impulse', 'Total impulse', 'N·s', 0],
    ['peak_thrust', 'Peak thrust', 'N', 0],
    ['isp', 'Specific impulse', 's', 1],
    ['burn_time', 'Burn time', 's', 2],
    ['max_pressure_psi', 'Peak pressure', 'psi', 0],
    ['initial_kn', 'Initial Kn', '', 0],
    ['peak_kn', 'Peak Kn', '', 0],
    ['mass_flux_lb', 'Peak mass flux', 'lb/in²s', 3],
    ['port_throat', 'Port/throat', '', 2],
    ['prop_mass', 'Propellant', 'kg', 3]
  ];

  function deltaTable(design, baseline) {
    if (!design) return '<p class="sub">Run the optimizer to compare.</p>';
    const rows = SPEC_ROWS.map(([key, label, unit, dp]) => {
      const a = baseline ? baseline[key] : null, b = design[key];
      if (b === undefined || b === null) return '';
      let delta = '';
      if (a) {
        const pct = (b / a - 1) * 100;
        const cls = Math.abs(pct) < 0.05 ? '' : (pct > 0 ? 'pos' : 'neg');
        delta = `<td class="n ${cls}">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</td>`;
      } else delta = '<td class="n"></td>';
      return `<tr><td>${label}</td>
        <td class="n">${a !== null && a !== undefined ? a.toFixed(dp) : '—'}</td>
        <td class="n" style="font-weight:650">${b.toFixed(dp)}</td>${delta}
        <td style="color:var(--ink-3)">${unit}</td></tr>`;
    }).join('');
    return `<div style="overflow-x:auto"><table class="data-table">
      <thead><tr><th></th><th class="n">yours</th><th class="n">optimized</th>
      <th class="n">Δ</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function marginRow(c, design) {
    const key = { max_pressure: 'max_pressure', peak_mass_flux: 'peak_mass_flux' }[c.metric] || c.metric;
    let value = design[key];
    if (value === undefined || value === null) return '';
    const limit = c.value;
    const ratio = c.op === '<=' ? value / limit : limit / Math.max(value, 1e-9);
    const pct = Math.max(0, Math.min(ratio, 1.35)) * 100;
    const cls = ratio > 1.0005 ? 'over' : (ratio > 0.97 ? 'close' : '');
    const shown = metricValue(design, c.metric);
    const limitShown = c.metric === 'max_pressure' || c.metric === 'avg_pressure'
      ? limit / 6894.757293168361
      : (c.metric === 'peak_mass_flux' ? limit / 703.0696 : limit);
    const dp = limitShown < 10 ? 3 : 0;
    return `<div class="margin-row">
      <span class="label">${c.label || metricLabel(c.metric)}</span>
      <span class="track"><span class="bar ${cls}" style="width:${Math.min(pct, 100).toFixed(1)}%"></span></span>
      <span class="value">${shown.toFixed(dp)} / ${limitShown.toFixed(dp)}</span></div>`;
  }

  function optionsTable(ctx) {
    const designs = ctx.designs || [];
    if (!designs.length) return '<p class="sub">No feasible designs found.</p>';
    const [ax, ay] = ctx.axes;
    const b = ctx.baseline;
    const rows = designs.map((d, i) => {
      const pct = v => b && b[v] ? ((d[v] / b[v] - 1) * 100) : null;
      const cell = v => {
        const p = pct(v);
        return p === null ? '<td class="n"></td>' :
          `<td class="n ${p >= 0 ? 'pos' : 'neg'}">${p >= 0 ? '+' : ''}${p.toFixed(2)}%</td>`;
      };
      return `<tr class="clickable ${i === ctx.selected ? 'pick' : ''}" data-index="${i}">
        <td>${d.designation || ('Option ' + (i + 1))}</td>
        <td class="n">${metricValue(d, ax).toFixed(0)}</td>${cell(ax)}
        <td class="n">${metricValue(d, ay).toFixed(0)}</td>${cell(ay)}
        <td class="n">${d.max_pressure_psi.toFixed(0)}</td>
        <td class="n">${d.peak_kn.toFixed(0)}</td>
        <td class="n">${d.mass_flux_lb.toFixed(3)}</td>
        <td><button class="chip" data-export="${i}">.ric</button></td></tr>`;
    }).join('');
    return `<div style="overflow-x:auto;max-height:340px"><table class="data-table">
      <thead><tr><th>class</th><th class="n">${metricLabel(ax)}</th><th class="n">Δ</th>
      <th class="n">${metricLabel(ay)}</th><th class="n">Δ</th>
      <th class="n">psi</th><th class="n">Kn</th><th class="n">flux</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  /* ------------------------------------------------- the search, live */

  const LIVE = { ghosts: [], lastSeed: -1 };

  function resetLive() { LIVE.ghosts = []; LIVE.lastSeed = -1; }

  function liveFrame(node, snap) {
    const [ax, ay] = snap.metrics;
    // A new seed starts somewhere else entirely; carrying its predecessor's
    // trail over would read as one search teleporting.
    if (snap.seed_index !== LIVE.lastSeed) { LIVE.ghosts = []; LIVE.lastSeed = snap.seed_index; }

    const feas = snap.points.filter(p => p[2]);
    const infeas = snap.points.filter(p => !p[2]);
    LIVE.ghosts.push(feas.map(p => [p[0], p[1]]));
    if (LIVE.ghosts.length > 14) LIVE.ghosts.shift();

    // Older generations fade out, so the population leaves a wake and you can
    // see which way the search is travelling.
    const trails = LIVE.ghosts.slice(0, -1).map((g, i) => ({
      x: g.map(p => p[0]), y: g.map(p => p[1]), mode: 'markers', type: 'scatter',
      marker: { size: 4, color: SERIES[0],
                opacity: 0.05 + 0.16 * (i / Math.max(LIVE.ghosts.length - 1, 1)) },
      hoverinfo: 'skip', showlegend: false
    }));

    const traces = trails.concat([
      { x: infeas.map(p => p[0]), y: infeas.map(p => p[1]), mode: 'markers',
        name: 'over a limit', type: 'scatter',
        marker: { size: 5, color: css('--ink-3'), opacity: .38 }, hoverinfo: 'skip' },
      { x: feas.map(p => p[0]), y: feas.map(p => p[1]), mode: 'markers',
        name: 'legal', type: 'scatter',
        marker: { size: 7, color: SERIES[0], opacity: .9,
                  line: { color: css('--surface'), width: 1 } }, hoverinfo: 'skip' },
      { x: snap.front.map(p => p[0]), y: snap.front.map(p => p[1]),
        mode: 'lines+markers', name: 'best so far', type: 'scatter',
        line: { color: SERIES[1], width: 2 },
        marker: { size: 7, color: SERIES[1], line: { color: css('--surface'), width: 1 } },
        hoverinfo: 'skip' }
    ]);

    Plotly.react(node, traces, Object.assign(theme(), {
      showlegend: true,
      transition: { duration: 320, easing: 'cubic-in-out' },
      margin: { l: 62, r: 18, t: 8, b: 44 },
      xaxis: Object.assign(theme().xaxis, { title: axisTitle(ax) }),
      yaxis: Object.assign(theme().yaxis, { title: axisTitle(ay) })
    }), { displayModeBar: false, responsive: true });
  }

  function liveSpark(node, trace) {
    if (!trace || trace.length < 2) { node.innerHTML = ''; return; }
    const running = [];
    let best = -Infinity;
    trace.forEach(t => { best = Math.max(best, t.a); running.push(best); });
    // Best-so-far only ever climbs, and it climbs within a narrow band. Filling
    // to zero would paint the whole box solid and hide every step.
    const lo = Math.min(...running), hi = Math.max(...running);
    const pad = Math.max((hi - lo) * 0.18, Math.abs(hi) * 0.004, 1e-6);
    Plotly.react(node, [{
      y: running, mode: 'lines', type: 'scatter',
      line: { color: SERIES[1], width: 2, shape: 'hv' }, hoverinfo: 'skip'
    }], Object.assign(theme(), {
      margin: { l: 0, r: 0, t: 6, b: 6 }, showlegend: false,
      xaxis: { visible: false },
      yaxis: { visible: false, range: [lo - pad, hi + pad] }
    }), { displayModeBar: false, responsive: true, staticPlot: true });
  }

  return { PANELS, PROFILES, theme, draw, metricValue, metricLabel, axisTitle,
           liveFrame, liveSpark, resetLive,
           crossSectionSVG, parallelSVG, deltaTable };
})();
