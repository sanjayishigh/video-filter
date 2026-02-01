function setMode(mode, element) {
    fetch(`/set_mode/${mode}`, { method: 'POST' });
    document.querySelectorAll('.filter-item').forEach(item => item.classList.remove('active'));
    element.classList.add('active');
}

function toggleFeature(feature, btnElement) {
    fetch(`/toggle/${feature}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.state) btnElement.classList.add('active');
            else btnElement.classList.remove('active');
        });
}

function snap() {
    const img = document.getElementById('stream');
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    
    const btn = document.querySelector('.shutter-btn');
    btn.style.opacity = 0.5;
    setTimeout(() => btn.style.opacity = 1, 150);

    const link = document.createElement('a');
    link.download = `camstudio_${Date.now()}.jpg`;
    link.href = canvas.toDataURL('image/jpeg');
    link.click();
}

function toggleFeature(feature, btn) {
    // 1. Send request
    fetch(`/toggle/${feature}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            // 2. Update Button Style based on server state
            if(data.state) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
}
