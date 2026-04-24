// Test API connection - paste this into browser console on https://talentranker-ai.onrender.com
async function testConnection() {
    console.log('Testing basic API connection...');

    try {
        const response = await fetch('/test');
        console.log('Test endpoint status:', response.status);
        const data = await response.json();
        console.log('Test endpoint response:', data);

        if (data.status === 'ok') {
            console.log('✅ Basic API connection working!');
            return true;
        } else {
            console.log('❌ Basic API connection failed:', data);
            return false;
        }
    } catch (error) {
        console.error('❌ Basic API connection error:', error);
        return false;
    }
}

async function testAPI() {
    console.log('Testing /rank endpoint...');

    // First test basic connection
    const connected = await testConnection();
    if (!connected) {
        console.log('❌ Cannot proceed with /rank test - basic connection failed');
        return;
    }

    try {
        const response = await fetch('/rank', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                resume: 'Experienced software engineer with 5 years of Python and JavaScript skills',
                jobs: ['Senior Software Engineer position requiring Python and JavaScript experience']
            })
        });

        console.log('Rank endpoint status:', response.status);
        const data = await response.json();
        console.log('Rank API Response:', data);

        if (data.match_score !== undefined) {
            console.log('✅ Rank API working! Match score:', data.match_score);
        } else {
            console.log('❌ Rank API returned unexpected response:', data);
        }
    } catch (error) {
        console.error('❌ Rank API Error:', error);
    }
}

// Test PDF upload
async function testPDFUpload() {
    console.log('Testing PDF upload...');

    try {
        const formData = new FormData();
        // Create a dummy file for testing
        const blob = new Blob(['%PDF-1.4 dummy content'], { type: 'application/pdf' });
        formData.append('file', blob, 'test.pdf');

        const response = await fetch('/rank', {
            method: 'POST',
            body: formData
        });

        console.log('PDF Response status:', response.status);
        const data = await response.json();
        console.log('PDF API Response:', data);
    } catch (error) {
        console.error('❌ PDF Upload Error:', error);
    }
}

console.log('API test functions loaded. Run testAPI() or testPDFUpload() to test.');
