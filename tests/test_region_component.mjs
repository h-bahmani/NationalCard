// Isolated event tests on synthetic objects; no browser, page, or user data.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(process.argv[2],'utf8');
const {default:render}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
globalThis.Image=class {naturalWidth=800;naturalHeight=500;complete=true;set src(v){this.onload?.();}};
globalThis.ResizeObserver=class {observe(){}disconnect(){}};
function mount(initial=[.1,.1,.9,.9]){
 const noop=()=>{};
 const context=new Proxy({}, {get:(o,k)=>o[k]??noop});
 const canvas={width:800,height:500,style:{},getContext:()=>context,
  getBoundingClientRect:()=>({left:0,top:0,width:400,height:250}),
  focus:noop,setPointerCapture:noop,hasPointerCapture:()=>true,releasePointerCapture:noop};
 const items={canvas,'.apply':{},'.status':{},'.hint':{},'.new':{},'.reset':{}};
 const states=[],events=[];
 const parentElement={querySelector:k=>items[k]};
 const cleanup=render({parentElement,data:{box:initial,initial,fields:[],active:'card',image:'synthetic',hint:'test'},
  setStateValue:(key,value)=>states.push({key,value}),setTriggerValue:(key,value)=>events.push({key,value})});
 return {canvas,items,states,events,cleanup};
}
const pointer=(x,y)=>({button:0,pointerId:1,clientX:x*400,clientY:y*250,preventDefault(){}});
const near=(actual,expected)=>actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-9,`${actual} != ${expected}`));
const a=mount();
a.items['.new'].onclick();a.canvas.onpointerdown(pointer(.2,.25));a.canvas.onpointermove(pointer(.6,.75));a.canvas.onpointerup(pointer(.6,.75));
near(a.states.at(-1).value,[.2,.25,.6,.75]);assert.equal(a.events.length,0);
a.items['.apply'].onclick();near(a.events[0].value.box,[.2,.25,.6,.75]);assert.equal(a.events[0].value.field,'card');
a.canvas.onpointerdown(pointer(.4,.5));a.canvas.onpointermove(pointer(.5,.6));a.canvas.onpointerup(pointer(.5,.6));
near(a.states.at(-1).value,[.3,.35,.7,.85]);
a.canvas.onpointerdown(pointer(.3,.35));a.canvas.onpointermove(pointer(.1,.2));a.canvas.onpointerup(pointer(.1,.2));
near(a.states.at(-1).value,[.1,.2,.7,.85]);
const before=a.states.length;
a.canvas.onpointerdown(pointer(.4,.5));a.canvas.onpointermove(pointer(.5,.6));a.canvas.onpointercancel(pointer(.5,.6));
assert.equal(a.states.length,before);a.items['.apply'].onclick();near(a.events.at(-1).value.box,[.1,.2,.7,.85]);
const b=mount([.2,.2,.8,.8]);b.items['.apply'].onclick();near(b.events[0].value.box,[.2,.2,.8,.8]);
a.items['.new'].onclick();a.canvas.onpointerdown(pointer(.4,.4));a.canvas.onpointerup(pointer(.4,.4));
assert.equal(a.items['.apply'].disabled,true);const n=a.events.length;a.items['.apply'].onclick();assert.equal(a.events.length,n);
a.cleanup();b.cleanup();
console.log('draw, commit, move, resize, cancel, isolated instances, empty crop: passed');
