## If you want to edit the client js files, do the following:
    
#### Do this once (make sure the ../human_aware_rl directory is not empty)
    sed -i 's/overcook\"/overcooked\"/g' ../human_aware_rl/overcooked_ai/overcooked_ai_js/package.json
    wget https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/37d14dd48ae93ad0363610a0a370221c47a79eb2/overcooked_ai_js/js/mdp.es6 -O ../human_aware_rl/overcooked_ai/overcooked_ai_js/js/mdp.es6
    wget https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/37d14dd48ae93ad0363610a0a370221c47a79eb2/overcooked_ai_js/js/task.es6 -O ../human_aware_rl/overcooked_ai/overcooked_ai_js/js/task.es6
        
    npm install
    npm link ../human_aware_rl/overcooked_ai/overcooked_ai_js/
    
#### Then each time you edit the files ( e.g. static/index.html , static/js/demo/js/overcooked-single.js ), run this, which crams all the js code into one file at static/js/demo/demo.js
    npm run build
