(() => {
  'use strict';
  const data = window.KU_PCPE;
  if (!data) return;
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const state = { classId: 'horizontal', case: 'typical', mask: 'aligned', budget: 'full' };
  const maskNames = { expert1: 'Expert 1', expert2: 'Expert 2', expert3: 'Expert 3', strict: 'Strict consensus', aligned: 'Aligned consensus' };
  const format = (number) => number.toFixed(3).replace(/^0/, '');
  const params = new URLSearchParams(location.search);
  if (Object.hasOwn(data.classes, params.get('class'))) state.classId = params.get('class');
  if (['typical', 'ambiguous'].includes(params.get('case'))) state.case = params.get('case');
  if (Object.hasOwn(maskNames, params.get('view'))) state.mask = params.get('view');

  $('#class-picker').innerHTML = Object.entries(data.classes).map(([key, item], index) => `
    <button class="class-button" type="button" data-class="${key}" aria-pressed="${key === state.classId}">
      <svg aria-hidden="true"><use href="#icon-${key}"/></svg><span>${item.name}</span><span class="class-number">${String(index + 1).padStart(2, '0')}</span>
    </button>`).join('');

  let renderSequence = 0;
  async function renderExample() {
    const sequence = ++renderSequence;
    const item = data.examples[`${state.classId}-${state.case}`];
    $$('.class-button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.class === state.classId)));
    $$('[data-case]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.case === state.case)));
    $$('[data-mask]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.mask === state.mask)));
    $('#comparison').setAttribute('aria-busy', 'true');
    const load = (src) => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = resolve;
      image.onerror = reject;
      image.src = src;
    });
    try {
      await Promise.all([load(item.image), load(item.overlays[state.mask])]);
      if (sequence !== renderSequence) return;
      const name = data.classes[state.classId].name;
      $('#example-image').src = item.image;
      $('#example-image').alt = `${name} composition: ${state.case} example from the KU-PCPE paper, image ${item.task.split('::')[1]}.`;
      $('#example-mask').src = item.overlays[state.mask];
      $('#mask-label').textContent = maskNames[state.mask];
      $('#example-title').textContent = `${name} composition`;
      $('#example-description').textContent = item.description;
      $('#example-agreement').textContent = format(item.pairDice);
      $('#example-task').textContent = `${state.case === 'typical' ? 'Typical' : 'Ambiguous'} · Image ${item.task.split('::')[1]}`;
      $('#comparison').setAttribute('aria-label', `${name} ${state.case} example, ${maskNames[state.mask]}`);
      $('#comparison').setAttribute('aria-busy', 'false');
    } catch {
      if (sequence !== renderSequence) return;
      $('#comparison').setAttribute('aria-busy', 'false');
      $('#example-description').textContent = 'This example could not be loaded. Try another class or read the examples in the supplement.';
    }
  }
  $('#class-picker').addEventListener('click', (event) => {
    const button = event.target.closest('[data-class]');
    if (!button) return;
    state.classId = button.dataset.class;
    renderExample();
  });
  $$('[data-case]').forEach((button) => button.addEventListener('click', () => {
    state.case = button.dataset.case;
    renderExample();
  }));
  $$('[data-mask]').forEach((button) => button.addEventListener('click', () => {
    state.mask = button.dataset.mask;
    renderExample();
  }));
  $('#reveal-slider').addEventListener('input', (event) => {
    const value = Number(event.target.value);
    $('#comparison').style.setProperty('--reveal', `${value}%`);
    event.target.setAttribute('aria-valuetext', `${value} percent of the photograph shown without annotation`);
  });
  function syncTeaserVisibility() {
    const visible = $('#hero-reveal').getAttribute('aria-pressed') === 'true';
    $('#hero-mask').hidden = !visible;
    $('#teaser-photo').alt = visible ? 'The same photograph, with regions selected by the experts shown in blue.' : 'The same photograph, with the annotation hidden.';
    $('#hero-reveal').textContent = visible ? 'Hide annotation' : 'Show annotation';
    $('#teaser-mask-caption').textContent = visible ? 'Consensus after alignment' : 'Photograph (annotation hidden)';
  }
  $('#hero-reveal').addEventListener('click', (event) => {
    const button = event.currentTarget;
    button.setAttribute('aria-pressed', String(button.getAttribute('aria-pressed') !== 'true'));
    syncTeaserVisibility();
  });
  let teaserSequence = 0;
  $$('[data-teaser-class]').forEach((button) => button.addEventListener('click', async () => {
    const sequence = ++teaserSequence;
    const classId = button.dataset.teaserClass;
    const item = data.examples[classId + '-typical'];
    const name = data.classes[classId].name;
    $('.teaser-pair').setAttribute('aria-busy', 'true');
    const load = (src) => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = resolve;
      image.onerror = reject;
      image.src = src;
    });
    try {
      await Promise.all([load(item.image), load(item.overlays.aligned)]);
      if (sequence !== teaserSequence) return;
      $('#teaser-original').src = item.image;
      $('#teaser-original').alt = `KU-PCP photograph queried for ${name} composition.`;
      $('#teaser-photo').src = item.image;
      $('#hero-mask').src = item.overlays.aligned;
      $('#teaser-class-name').textContent = name;
      $$('[data-teaser-class]').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
      syncTeaserVisibility();
      $('.teaser-pair').setAttribute('aria-busy', 'false');
    } catch {
      if (sequence !== teaserSequence) return;
      $('.teaser-pair').setAttribute('aria-busy', 'false');
      $('#teaser-mask-caption').textContent = 'Example unavailable. Explore the annotations below.';
    }
  }));

  const maxClassTasks = Math.max(...Object.values(data.classes).map((item) => item.train + item.test));
  $('#class-distribution').innerHTML = Object.values(data.classes).map((item) => `
    <div class="distribution-item"><div><span>${item.name}</span><span>${item.train} / ${item.test}</span></div>
      <div class="distribution-track" role="img" aria-label="${item.name}: ${item.train} training and ${item.test} test tasks"><span style="width:${item.train / maxClassTasks * 100}%"></span><span style="width:${item.test / maxClassTasks * 100}%"></span></div>
    </div>`).join('');

  function renderChart() {
    const rows = data.benchmarks[state.budget];
    $('#chart-bars').innerHTML = rows.map((item) => `
      <div class="chart-row ${item.kind}" role="listitem" title="95% confidence interval from resampling tasks: ${format(item.low)} to ${format(item.high)}">
        <span>${item.label === 'Train class template' ? 'Template from training' : item.label}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--score:${item.dice * 100}%"></span><span class="bar-ci" style="--low:${item.low * 100}%;--interval:${(item.high - item.low) * 100}%"></span></span><span>${format(item.dice)}</span><span class="sr-only">95% confidence interval from resampling tasks: ${format(item.low)} to ${format(item.high)}.</span>
      </div>`).join('');
    $('#chart-bars').setAttribute('aria-label', `${state.budget === 'full' ? 'Full-bank' : 'Matched-12'} benchmark Dice scores`);
    $('#chart-note').textContent = state.budget === 'full'
      ? 'Model scores use the consensus of three experts after accounting for complementary selections. The human reference compares one expert with the consensus of the other two, after alignment. The greedy oracle uses the target mask to select up to three proposals. Whiskers show 95% confidence intervals obtained by resampling tasks.'
      : 'Each selector receives the same first 12 proposals from SAM2.1 and can combine up to three. The exact oracle checks every combination of up to three proposals. Both oracles use the target mask; the template method and Qwen do not. Whiskers show 95% confidence intervals obtained by resampling tasks.';
    $$('[data-budget]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.budget === state.budget)));
    $('.chart-legend .human-key').parentElement.hidden = state.budget !== 'full';
  }
  $$('[data-budget]').forEach((button) => button.addEventListener('click', () => {
    state.budget = button.dataset.budget;
    renderChart();
  }));

  $('#copy-citation').addEventListener('click', async () => {
    const text = $('#citation-code').textContent;
    const label = $('#copy-citation span');
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = 'Copied!';
      $('#copy-status').textContent = 'BibTeX citation copied to clipboard.';
      setTimeout(() => { label.textContent = 'Copy BibTeX'; }, 2500);
    } catch {
      const range = document.createRange();
      range.selectNodeContents($('#citation-code'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = 'Text selected';
      $('#copy-status').textContent = 'Citation selected. Copy it with your keyboard, or use Download BibTeX.';
    }
  });
  renderExample();
  renderChart();
})();
