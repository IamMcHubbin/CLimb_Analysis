const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function app() {
  const elements = new Map(), requests = [], animation = [];
  function element(id) {
    if (elements.has(id)) return elements.get(id);
    const calls = [];
    const drawing = Object.fromEntries(['clearRect','beginPath','moveTo','lineTo','stroke','arc','fill',
      'fillRect','setTransform'].map(name => [name, (...args) => calls.push([name,...args])]));
    const value = {id, style:{}, hidden:false, checked:false, value:'heavy', clientWidth:200,
      clientHeight:400, width:0, height:0, currentTime:0, events:{}, calls, children:[],
      classList:{toggle(){},add(){},remove(){}},
      addEventListener(name, handler){this.events[name] = handler;},
      getContext(){return drawing;}, querySelectorAll(){return [];},
      replaceChildren(){this.children = [];}, appendChild(child){this.children.push(child);},
      removeAttribute(name){delete this[name];}, pause(){this.paused=true;}, load(){},
      scrollIntoView(){}, getBoundingClientRect(){return {width:200,height:100,left:0,top:0};}};
    elements.set(id, value);
    return value;
  }
  const config = {default_pose_model:'heavy',max_upload_bytes:1000,target_fps:30,max_long_edge:1280,
    pose_models:[{id:'heavy',label:'Heavy',available:true},{id:'vitpose',label:'ViTPose',available:true}]};
  const context = vm.createContext({document:{getElementById:element,documentElement:{},
    createElement:()=>({...element('new'+Math.random()),children:[]}),addEventListener(){}},
    getComputedStyle:()=>({getPropertyValue:()=> '#123456'}),
    window:{devicePixelRatio:1,addEventListener(){}}, console,URLSearchParams,
    requestAnimationFrame:fn=>animation.push(fn),clearInterval(){},setInterval(){},setTimeout,
    fetch:async url=>{requests.push(url);return {ok:true,status:200,json:async()=>url==='/config'?config:[]};}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'),context);
  return {context,element,requests,animation,run:code=>vm.runInContext(code,context)};
}

test('both canvases use identical frame, smoothing, and edges; gaps clear both', async()=>{
  const a=app();
  await new Promise(setImmediate);
  a.run(`state.video={has_footage:true}; state.keypoints={fps:30,frame_count:3,
    frames:[[[.1,.2,.9],[.4,.6,.8]],null,[[.2,.3,.9],[.5,.7,.8]]],
    landmark_connections:[[0,1]],match_iou:[.8,null,.9]};
    state.smoothed=[[[.15,.25,.9],[.45,.65,.8]],null,[[.25,.35,.9],[.55,.75,.8]]];
    drawPlayhead=()=>{}; drawOverlay();`);
  assert.deepEqual(a.element('overlay').calls,a.element('skeleton-only').calls);
  assert.ok(a.element('overlay').calls.some(c=>c[0]==='moveTo'&&c[1]===20&&c[2]===80));
  for (const name of ['overlay','skeleton-only']) a.element(name).calls.length=0;
  a.element('player').currentTime=1/30+.001;
  a.run('drawOverlay()');
  assert.deepEqual(a.element('overlay').calls,[['clearRect',0,0,200,400]]);
  assert.deepEqual(a.element('overlay').calls,a.element('skeleton-only').calls);
  a.element('player').currentTime=2/30+.001;
  a.element('smooth').checked=true;
  a.run('drawOverlay()');
  assert.deepEqual(a.element('overlay').calls,a.element('skeleton-only').calls);
  assert.ok(a.element('overlay').calls.some(c=>c[0]==='moveTo'&&c[1]===50&&c[2]===140));
  assert.equal(a.animation.length,1); // no extra animation loops on drawing/run refresh
});

test('opening picker and changing model perform no inference and invalidate selection',async()=>{
  const a=app(); await new Promise(setImmediate);
  a.run(`state.video={id:'v'}; prepareCandidates();`);
  a.run(`state.candidates={selection_id:'old'};state.selected=0;state.frameReady=true;`);
  a.element('pose-model').value='vitpose';
  a.element('pose-model').events.change();
  assert.equal(a.run('state.candidates'),null);
  assert.equal(a.run('state.selected'),null);
  assert.equal(a.element('analyse').disabled,true);
  assert.equal(a.element('frame-wrap').hidden,true);
  assert.equal(a.requests.some(p=>p.includes('/candidates')),false);
});

test('late detection response cannot restore selection after model change',async()=>{
  const a=app(); await new Promise(setImmediate);
  let resolve;
  a.context.fetch=url=>{a.requests.push(url);return new Promise(done=>{resolve=done;});};
  a.run(`state.video={id:'v'};`);
  const pending=a.run('loadCandidates()');
  a.element('pose-model').value='vitpose';
  a.element('pose-model').events.change();
  resolve({ok:true,status:200,json:async()=>({selection_id:'old',pose_model:'heavy'})});
  await pending;
  assert.equal(a.run('state.candidates'),null);
  assert.equal(a.element('analyse').disabled,true);
});

test('submission sends the displayed selection and model, not a global default',async()=>{
  const a=app(); await new Promise(setImmediate);
  const posts=[];
  a.context.fetch=async(url,options)=>{posts.push(JSON.parse(options.body));return {
    ok:true,status:202,json:async()=>({id:'job',analysis_run_id:'run'})};};
  a.run(`state.video={id:'v'};state.candidates={pose_model:'vitpose',selection_id:'exact'};
    state.selected=1;state.frameReady=true;pollJob=()=>{};refreshAnalysisRuns=async()=>{};`);
  await a.element('analyse').events.click();
  assert.deepEqual(posts,[{candidate_index:1,pose_model:'vitpose',selection_id:'exact'}]);
});

test('playback layout always specifies two equal columns, never a stacking breakpoint',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../static/index.html'),'utf8');
  assert.match(html,/\.playback-grid\s*\{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
  assert.equal((html.match(/grid-template-columns/g)||[]).length,1);
  assert.ok(html.indexOf('id="player"')<html.indexOf('id="skeleton-only"'));
});
