'use strict';
(() => {
  const pairs = window.SHOWCASE_DATA?.pairs;
  if (!pairs?.length) return;
  const get = (id) => document.getElementById(id);
  get('pair-select').addEventListener('change', (event) => {
    const pair = pairs.find((item) => item.id === event.target.value);
    if (!pair) return;
    get('matches-image').src = pair.matches_image;
    get('matches-link').href = pair.matches_image;
    get('matches-image').alt = `${pair.scene} ${pair.source_frame} to ${pair.target_frame}: numbered corresponding points`;
    get('pair-caption').textContent = `${pair.scene} ${pair.source_frame} → ${pair.target_frame} · ${pair.num_pairs} point pairs. Click the image to enlarge.`;
    get('matrix').textContent = pair.H.map((row) => '[ ' + row.map((value) => value.toExponential(6).padStart(14)).join(' ') + ' ]').join('\n');
    get('system').textContent = pair.A.map((row, i) => '[ ' + [...row, pair.b[i]].map((value) => value.toPrecision(7).padStart(15)).join(' ') + ' ]').join('\n');
    get('system-link').href = pair.system_path;
    get('matrix-link').href = pair.json_path;
  });
})();

(() => {
  const select = document.getElementById('ransac-select');
  if (!select) return;
  select.addEventListener('change', () => {
    for (const panel of document.querySelectorAll('[data-ransac-pair]')) {
      panel.hidden = panel.dataset.ransacPair !== select.value;
    }
  });
})();

(() => {
  const select = document.getElementById('match-select');
  if (!select) return;
  select.addEventListener('change', () => {
    for (const panel of document.querySelectorAll('[data-match-pair]')) {
      panel.hidden = panel.dataset.matchPair !== select.value;
    }
  });
})();

(() => {
  const select = document.getElementById('descriptor-select');
  if (!select) return;
  select.addEventListener('change', () => {
    for (const panel of document.querySelectorAll('[data-descriptor-image]')) {
      panel.hidden = panel.dataset.descriptorImage !== select.value;
    }
  });
})();

(() => {
  const examples = window.B1_DATA?.examples;
  const get = (id) => document.getElementById(id);
  if (!examples?.length || !get('corner-select')) return;
  get('corner-select').addEventListener('change', (event) => {
    const result = examples.find((item) => item.id === event.target.value);
    if (!result) return;
    for (const [prefix, key] of [['harris-top', 'harris_top'], ['anms', 'anms_image'], ['harris-all', 'harris_all']]) {
      get(`${prefix}-image`).src = result[key];
      get(`${prefix}-link`).href = result[key];
      get(`${prefix}-image`).width = result.size[0];
      get(`${prefix}-image`).height = result.size[1];
      get(`${prefix}-image`).alt = `${result.title}: ${prefix}`;
    }
    get('harris-top-caption').textContent = `${result.title} · strongest ${result.selected_count} Harris corners · no ANMS`;
    get('anms-caption').textContent = `${result.title} · ANMS ${result.selected_count} corners`;
    get('harris-all-caption').textContent = `${result.title} · all ${result.candidate_count} Harris candidates`;
    get('corner-csv-link').href = result.csv;
    get('corner-record-link').href = result.record_path;
    get('corner-npz-link').href = result.features;
  });
})();
