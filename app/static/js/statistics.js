const main = document.querySelector('main');





document.addEventListener('internalRendered', (e) => {
    if (e.detail.pageName != 'statistics') return;

    const page = main.querySelector('.page-container');
    form
    const plotGrid = page.querySelector('.plot-grid');
    const testPlotContainer = plotGrid.querySelector('#test-plot');

    async function loadStatPlots() {
        // Update UI
        plotGrid.classList.add('waiting');
        // Build request params (filters?)
        const params = new URLSearchParams();
        params.append('date', ['2026-01-01', '2026-01-28']); // filter exemple
        // Request plots
        const response = await fetch(`/ajax/stat_plots?${params}`);
        if (!response.ok) {
            plotGrid.classList.remove('waiting');
            plotGrid.classList.add('error');
            return
        }
        const content = await response.json();
        plotGrid.classList.remove('waiting');
        // Extract data
        const testFig = JSON.parse(content.test);
        await Plotly.react(testPlotContainer, testFig.data, testFig.layout, { responsive: true });
    }

    form.addEventListener('submit', (e) => {

    })


    function relayoutPlots() {
        Plotly.relayout(testPlotContainer, {
            width: testPlotContainer.clientWidth,
            height: testPlotContainer.clientHeight
        });
    }

    // Relayout plots on page resize
    const observer = new ResizeObserver(() => {
        relayoutPlots();
    });

    // Init plots on page load
    loadStatPlots().then(() => {
        observer.observe(page);
    });
});