const main = document.querySelector('main');





document.addEventListener('internalRendered', (e) => {
    if (e.detail.pageName != 'statistics') return;

    const page = main.querySelector('.page-container .stats');
    const barPlotContainer = page.querySelector('#bar-plot');
    const linePlotContainer = page.querySelector('#line-plot');
    const piePlotContainer = page.querySelector('#pie-plot');

    async function loadAivalableModels() {

    }

    async function loadAvailableMetrics() {

    }

    async function loadKpis() {
        
    }

    async function loadPlots() {
        // Update UI
        page.classList.add('waiting');
        // Build request params (filters?)
        const params = new URLSearchParams();
        params.append('date', ['2026-01-01', '2026-01-28']); // filter exemple
        // Request plots
        const response = await fetch(`/ajax/stat_plots?${params}`);
        if (!response.ok) {
            page.classList.remove('waiting');
            page.classList.add('error');
            return
        }
        // Extract data
        const content = await response.json();
        const barFig = JSON.parse(content.bar);
        const lineFig = JSON.parse(content.line);
        const pieFig = JSON.parse(content.pie);
        // Render plot
        page.classList.remove('waiting');
        await Plotly.react(barPlotContainer, barFig.data, barFig.layout, { responsive: true });
        await Plotly.react(linePlotContainer, lineFig.data, lineFig.layout, { responsive: true });
        await Plotly.react(piePlotContainer, pieFig.data, pieFig.layout, { responsive: true });
    }


    function relayoutPlots() {
        Plotly.relayout(barPlotContainer, {
            width: barPlotContainer.clientWidth,
            height: barPlotContainer.clientHeight
        });
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

    // Init plots on page load
    loadPlots().then(() => {
        observer.observe(page);
    });
});