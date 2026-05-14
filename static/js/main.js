let currentPolymer = null;
let currentCurveData = null;

function showMessage(msg, type = 'info') {
    const msgDiv = document.getElementById('messages');
    msgDiv.innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
        ${msg}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>`;
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(a => a.remove());
    }, 5000);
}

// ========== SIDEBAR TOGGLE (robust, no duplicate listeners) ==========
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebarToggleBtn');
    const showBtn = document.getElementById('showSidebarBtn');

    function hideSidebar() {
        sidebar.classList.add('hidden');
        if (showBtn) showBtn.style.display = 'flex';
    }
    function showSidebar() {
        sidebar.classList.remove('hidden');
        if (showBtn) showBtn.style.display = 'none';
    }

    if (toggleBtn) toggleBtn.addEventListener('click', hideSidebar);
    if (showBtn) showBtn.addEventListener('click', showSidebar);

    // Ensure floating button hidden initially
    if (showBtn && !sidebar.classList.contains('hidden')) {
        showBtn.style.display = 'none';
    }

    // Video autoplay (ignore if blocked)
    const video = document.querySelector('.video-banner video');
    if (video) video.play().catch(e => console.log('Video autoplay prevented:', e));
});
// Video mute/unmute functionality
const videoEl = document.getElementById('topVideo');
const muteBtn = document.getElementById('muteToggleBtn');

if (videoEl && muteBtn) {
    function updateMuteIcon() {
        const icon = muteBtn.querySelector('i');
        if (videoEl.muted) {
            icon.className = 'fas fa-volume-mute';
        } else {
            icon.className = 'fas fa-volume-up';
        }
    }
    muteBtn.addEventListener('click', () => {
        videoEl.muted = !videoEl.muted;
        updateMuteIcon();
    });
    // initial icon state (video starts muted)
    updateMuteIcon();
}
// ========== FILE UPLOAD ==========
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput.files.length) {
        showMessage('Please select a CSV file', 'warning');
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    showMessage('Uploading...', 'info');
    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            showMessage('File uploaded successfully!', 'success');
            document.getElementById('previewSection').style.display = 'block';
            let html = '<table class="table table-dark table-striped table-smaller"><thead><tr>';
            Object.keys(data.preview[0]).forEach(k => html += `<th>${k}</th>`);
            html += '</thead><tbody>';
            data.preview.forEach(row => {
                html += '<tr>';
                Object.values(row).forEach(v => html += `<td>${v}<\/td>`);
                html += '</tr>';
            });
            html += '</tbody></table>';
            document.getElementById('previewTable').innerHTML = html;
            document.getElementById('trainSection').style.display = 'block';
        } else {
            showMessage(data.error, 'danger');
        }
    } catch (err) {
        showMessage('Upload failed', 'danger');
    }
});

// ========== TRAIN MODEL ==========
document.getElementById('trainBtn').addEventListener('click', async () => {
    document.getElementById('trainProgress').innerHTML = '<div class="spinner-border text-success"></div> Training...';
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showMessage('Model trained successfully!', 'success');
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('analysisSection').style.display = 'block';

            const m = data.metrics;
            document.getElementById('metrics').innerHTML = `
                <p>Dataset size: ${m.n_samples}</p>
                <p>Random Forest R²: ${m.rf_r2.toFixed(4)}</p>
                <p>SVM R²: ${m.svm_r2 ? m.svm_r2.toFixed(4) : 'N/A'}</p>
                <p>LightGBM R²: ${m.lgbm_r2 ? m.lgbm_r2.toFixed(4) : 'N/A'}</p>
                <p>Final Ensemble R²: ${m.final_r2.toFixed(4)}</p>
                <p>MAE: ${m.final_mae.toFixed(4)}</p>
                <p>RMSE: ${m.final_rmse.toFixed(4)}</p>`;

            // R² Bar Chart
            const models = ['Random Forest'];
            const r2s = [m.rf_r2];
            if (m.svm_r2) { models.push('SVM'); r2s.push(m.svm_r2); }
            if (m.lgbm_r2) { models.push('LightGBM'); r2s.push(m.lgbm_r2); }
            models.push('Final Ensemble'); r2s.push(m.final_r2);
            const barData = [{
                x: models, y: r2s, type: 'bar', marker: { color: '#00cc99' },
                text: r2s.map(v => (v*100).toFixed(1)+'%'),
                textposition: 'outside'
            }];
            const layout = { title: 'R² Comparison', yaxis: { range: [0,1], tickformat: '.0%' }, width: 700, height: 450 };
            Plotly.newPlot('r2Chart', barData, layout);

            const predictions = data.predictions_preview;

            // Table 1: Parameters
            let paramsHtml = `<h4>Parameters responsible for degradation, predicted % and confidence:</h4>
                <div class="table-responsive"><table class="table table-dark table-striped table-smaller">
                <thead><tr><th>ID</th><th>Shape</th><th>Polymer</th><th>Polymer_name</th>
                <th>Degradation</th><th>shoot_mg</th><th>root_mg</th>
                <th>Final Degradation (%)</th><th>Confidence (%)</th></tr></thead><tbody>`;
            predictions.forEach(p => {
                paramsHtml += `<tr>
                    <td>${p.ID || ''}</td><td>${p.Shape || ''}</td><td>${p.Polymer || ''}</td>
                    <td>${p.Polymer_name || ''}</td><td>${p.Degradation || ''}</td>
                    <td>${p.shoot_mg || ''}</td><td>${p.root_mg || ''}</td>
                    <td>${p['Final Degradation (%)']}</td><td>${p['Confidence (%)'] || ''}</td>
                </tr>`;
            });
            paramsHtml += `</tbody></table></div>`;
            document.getElementById('parametersTable').innerHTML = paramsHtml;

            // Table 2: Ranking
            let sorted = [...predictions].sort((a,b) => (a['Final Degradation (%)']||0) - (b['Final Degradation (%)']||0));
            let rankingHtml = `<h4>Polymers Ranked by Recyclability (based on degradation)</h4>
                <div class="table-responsive"><table class="table table-dark table-smaller">
                <thead><tr><th>Polymer</th><th>Final Degradation (%)</th><th>Degradation Rank</th>
                <th>Recyclability (%)</th><th>Recyclability Class</th></tr></thead><tbody>`;
            sorted.forEach((p, idx) => {
                rankingHtml += `<tr>
                    <td>${p.Polymer || p.Polymer_name || p.ID}</td>
                    <td>${p['Final Degradation (%)']}</td><td>${idx+1}</td>
                    <td>${p['Recyclability (%)']}</td><td>${p['Recyclability Class']}</td>
                </tr>`;
            });
            rankingHtml += `</tbody></table></div>`;
            document.getElementById('rankingTable').innerHTML = rankingHtml;

            // Populate polymer dropdown
            const polyRes = await fetch('/api/polymer_list');
            const polyData = await polyRes.json();
            const select = document.getElementById('polymerSelect');
            select.innerHTML = '<option value="">Select polymer</option>';
            polyData.polymers.forEach(p => {
                select.innerHTML += `<option value="${p}">${p}</option>`;
            });
        } else {
            showMessage(data.error, 'danger');
        }
    } catch (err) {
        showMessage('Training failed: ' + err.message, 'danger');
    } finally {
        document.getElementById('trainProgress').innerHTML = '';
    }
});

// ========== LOAD POLYMER DETAILS ==========
document.getElementById('loadPolymerBtn').addEventListener('click', async () => {
    const polymerId = document.getElementById('polymerSelect').value;
    if (!polymerId) return;
    currentPolymer = polymerId;
    const res = await fetch('/api/polymer_details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polymer_id: polymerId })
    });
    const data = await res.json();
    if (res.ok) {
        document.getElementById('polymerDetails').style.display = 'block';
        document.getElementById('polymerName').innerHTML = `${data.polymer} (${data.info.full_name})`;
        document.getElementById('polymerInfo').innerHTML = `<strong>Type:</strong> ${data.info.type}<br>
            <strong>Examples:</strong> ${data.info.examples.join(', ')}<br>
            <strong>Uses:</strong> ${data.info.uses.join(', ')}`;
        let imgHtml = '';
        if (data.images.sample) imgHtml += `<div class="col-md-4"><img src="${data.images.sample}" class="img-fluid" alt="Sample"></div>`;
        if (data.images.structure) imgHtml += `<div class="col-md-4"><img src="${data.images.structure}" class="img-fluid" alt="Structure"></div>`;
        if (data.images.degradation) imgHtml += `<div class="col-md-4"><img src="${data.images.degradation}" class="img-fluid" alt="UV Degradation"></div>`;
        document.getElementById('polymerImages').innerHTML = imgHtml;
        document.getElementById('inputParams').innerHTML = JSON.stringify(data.input_params, null, 2);
        document.getElementById('degValue').innerText = data.degradation;
        document.getElementById('recyclabilityValue').innerText = data.recyclability;
        document.getElementById('recyclabilityClass').innerText = data.recyclability_class;
        document.getElementById('confidenceValue').innerText = data.confidence;

        // Curve
        const curveRes = await fetch('/api/degradation_curve', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ polymer_id: polymerId })
        });
        const curveData = await curveRes.json();
        currentCurveData = curveData;
        const trace = { x: curveData.days, y: curveData.degradation, mode: 'lines', name: 'Degradation %' };
        Plotly.newPlot('curveChart', [trace], { title: 'Degradation Curve', xaxis: { title: 'Days' }, yaxis: { title: 'Degradation (%)' } });

        updateTimeSlider();
        updateTargetSlider();
    } else {
        showMessage('Polymer not found', 'danger');
    }
});

async function updateTimeSlider() {
    const time = document.getElementById('timeSlider').value;
    document.getElementById('timeValue').innerText = time;
    if (!currentPolymer) return;
    const res = await fetch('/api/recyclability_at_time', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polymer_id: currentPolymer, time_days: parseFloat(time) })
    });
    const data = await res.json();
    document.getElementById('timeDegradation').innerText = data.degradation_percent;
    document.getElementById('timeRecyclability').innerText = data.recyclability_percent;
}
async function updateTargetSlider() {
    const target = document.getElementById('targetSlider').value;
    document.getElementById('targetValue').innerText = target;
    if (!currentPolymer) return;
    const res = await fetch('/api/time_to_degradation', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polymer_id: currentPolymer, target_percent: parseFloat(target) })
    });
    const data = await res.json();
    document.getElementById('targetDays').innerText = data.days.toFixed(2);
    if (currentCurveData) {
        const idx = currentCurveData.degradation.findIndex(d => d >= target);
        const degAtTarget = idx !== -1 ? currentCurveData.degradation[idx] : target;
        const recyclability = (1 - degAtTarget/100) * 100 * 0.85;
        document.getElementById('targetRecyclability').innerText = Math.max(0, Math.min(100, recyclability)).toFixed(2);
    }
}

document.getElementById('timeSlider').addEventListener('input', updateTimeSlider);
document.getElementById('targetSlider').addEventListener('input', updateTargetSlider);

// Background selector
document.getElementById('bgSelect').addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'none') document.body.style.background = '#0a2f1f';
    else document.body.style.background = `url('/static/background_images/${val}.jpg') center/cover fixed`;
});
//background images should be placed in static/background_images/ with names matching the select options (e.g. 'nature.jpg', 'abstract.jpg', etc.)

    document.addEventListener('DOMContentLoaded', function() {
        const selectEl = document.getElementById('bgSelect');
        if (!selectEl) return;

        selectEl.addEventListener('change', function() {
            let selected = this.value;
            if (selected === 'none') {
                // Restore your original gradient
                document.body.style.background = "radial-gradient(circle at 20% 30%, #0a0f1a, #03070f)";
                document.body.style.backgroundImage = "";
            } else {
                let filename = decodeURIComponent(selected);
                // Path now inside static folder
                let imageUrl = `/static/background_images/${filename}`;
                document.body.style.background = `url('${imageUrl}') no-repeat center center fixed`;
                document.body.style.backgroundSize = "cover";
            }
        });
    });