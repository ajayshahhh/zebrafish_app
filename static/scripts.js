$(document).ready(function() {
    $('#uploadForm').submit(function() {
        $('#loadingOverlay').show();  // Show the loader overlay
    });
});

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'block';
}