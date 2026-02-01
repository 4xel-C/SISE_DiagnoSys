const main = document.querySelector('main');





document.addEventListener('internalRendered', (e) => {
    if (e.detail.pageName != 'statistics') return;

    const page = main.querySelector('.page-container');
    const statsForm = page.querySelector('form#stats');
    const linePlotContainer = statsForm.querySelector('#line-plot');
    const piePlotContainer = statsForm.querySelector('#pie-plot');


    async function loadAvailableModels() {
        // Request available models
        const response = await fetch('ajax/get_recorded_models');
        const content = await response.json();
        // Update model options
        const modelSelect = statsForm.elements.model;
        content.models.forEach(model => {
            const opt = document.createElement('option');
            opt.name = model;
            opt.textContent = model;
            modelSelect.appendChild(opt);
        })
    }

    async function loadKpis(params) {
        const kpisList = statsForm.querySelector('.kpis');
        // Request KPIs
        const response = await fetch(`ajax/get_kpis?${params}`);
        if (!response.ok) {
            showError("Impossible de charger les KPIs");
            return
        }
        // Render kpis
        const content = await response.json();
        kpisList.querySelectorAll('li').forEach(kpiElement => {
            const kpiValue = kpiElement.querySelector('.value');
            const metricName = kpiElement.dataset.metricName;
            kpiValue.textContent = content[metricName];
        })
    }

    async function loadPlots(params) {
        // Update UI
        statsForm.classList.add('waiting');
        // Request plots
        const response = await fetch(`/ajax/stat_plots?${params}`);
        if (!response.ok) {
            showError("Impossible de charger les statistiques");
            return
        }
        // Extract data
        const content = await response.json();
        // Render plot
        statsForm.classList.remove('waiting');
        await Plotly.react(linePlotContainer, content.line.data, content.line.layout, { responsive: true });
        await Plotly.react(piePlotContainer, content.pie.data, content.pie.layout, { responsive: true });
    }

    function relayoutPlots() {
        Plotly.relayout(linePlotContainer, {
            width: linePlotContainer.clientWidth,
            height: linePlotContainer.clientHeight
        });
        Plotly.relayout(piePlotContainer, {
            width: piePlotContainer.clientWidth,
            height: piePlotContainer.clientHeight
        });
    }

    // Relayout plots on page resize
    const observer = new ResizeObserver(() => {
        relayoutPlots();
    });

    // On new metric selected
    statsForm.querySelectorAll('.metric .radio').forEach(radio => {
        radio.addEventListener('click', () => {
            const previous = statsForm.querySelector('.radio.checked');
            if (previous) {
                previous.classList.remove('checked');
            }
            radio.classList.add('checked');
            radio.querySelector('input[type="radio"]').checked = true;
            radio.dispatchEvent(new Event('change', { bubbles: true }));
        })
    })

    // On form change
    statsForm.addEventListener('change', (e) => {
        const params = new URLSearchParams(new FormData(statsForm));
        if (e.target.name != 'model') {
            loadKpis(params);
        }
        loadPlots(params).then(() => {
            relayoutPlots();
        })
    })

    // Init on page load
    const p = new URLSearchParams(new FormData(statsForm));
    loadAvailableModels();
    loadKpis(p);
    loadPlots(p).then(() => {
        observer.observe(page);
    });
});