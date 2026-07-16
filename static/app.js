const $=id=>document.getElementById(id), canvas=$("canvas"),ctx=canvas.getContext("2d");
const camera=document.createElement("video"); camera.autoplay=true; camera.muted=true; camera.playsInline=true;
let running=false,stream=null,requestInFlight=false;

function draw(d){
  ctx.fillStyle="#0b1220";ctx.fillRect(0,0,640,480);
  if(camera.readyState>=2)ctx.drawImage(camera,0,0,640,480);
  for(const o of d.objects||[]){const [x1,y1,x2,y2]=o.bbox;ctx.strokeStyle=o.class_name==="person"?"#38bdf8":"#facc15";ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`${o.class_name} #${o.track_id||"-"} ${(o.confidence*100).toFixed(0)}%`,x1,y1-4)}
  for(const p of d.poses||[]){for(const k of Object.values(p.keypoints)){ctx.fillStyle="#f472b6";ctx.beginPath();ctx.arc(k.x*640,k.y*480,4,0,7);ctx.fill()}}
  $("current").textContent=JSON.stringify(d.current_step,null,2);$("vlm-result").textContent=JSON.stringify(d.vlm_result,null,2);$("events").textContent=JSON.stringify(d.recent_events,null,2);$("metrics").textContent=`FPS: ${d.fps} / 推論: ${d.inference_ms}ms / VLM: ${d.vlm_calls||0}`;
}
async function call(path,opts={}){const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.json()}
function stopCamera(){if(stream){stream.getTracks().forEach(track=>track.stop());stream=null}camera.srcObject=null}
function scheduleFrame(){if(running)setTimeout(sendCameraFrame,150)}
async function sendCameraFrame(){
  if(!running||requestInFlight)return scheduleFrame();
  if(camera.readyState<2)return scheduleFrame();
  requestInFlight=true;
  try{
    const blob=await new Promise(resolve=>{const capture=document.createElement("canvas");capture.width=640;capture.height=480;capture.getContext("2d").drawImage(camera,0,0,640,480);capture.toBlob(resolve,"image/jpeg",0.85)});
    if(!blob)throw new Error("カメラフレームを作成できませんでした");
    const form=new FormData();form.append("file",blob,"camera.jpg");draw(await call("/api/analyze/image",{method:"POST",body:form}));
  }catch(error){running=false;$("metrics").textContent=`カメラエラー: ${error.message}`;stopCamera()}
  finally{requestInFlight=false;scheduleFrame()}
}
$("start").onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});camera.srcObject=stream;await camera.play();
    await call("/api/session/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sop_id:$("sop").value,source_type:"camera",source_name:"browser camera"})});
    running=true;sendCameraFrame();
  }catch(error){stopCamera();$("metrics").textContent=`カメラを開始できません: ${error.message}`}
};
$("pause").onclick=()=>{running=false;stopCamera()};
$("stop").onclick=async()=>{running=false;stopCamera();await call("/api/session/stop",{method:"POST"})};
$("reset").onclick=async()=>{running=false;stopCamera();await call("/api/session/stop",{method:"POST"});await $("start").onclick()};
$("vlm").onclick=async()=>draw(await call("/api/vlm/analyze",{method:"POST"}));
$("upload").onclick=()=>$("video").click();
$("video").onchange=async e=>{const f=e.target.files[0];if(!f)return;running=false;stopCamera();const form=new FormData();form.append("file",f);const r=await call("/api/analyze/video",{method:"POST",body:form});draw(r.last)};
