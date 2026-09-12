import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ---- costanti condivise con il backend (bar_env.py) ----
const ITEMS = ["caffe", "cappuccino", "cornetto_semplice", "cornetto_nutella"];
const ITEM_LABELS = { caffe: "caffe", cappuccino: "cappuccino", cornetto_semplice: "cornetto semplice", cornetto_nutella: "cornetto alla nutella" };
const MAX_COUNT = 3;
const FOOD_MODEL = { caffe: "cup-coffee", cappuccino: "mug", cornetto_semplice: "croissant", cornetto_nutella: "croissant" };
const CHARACTERS = ["character-male-a", "character-male-b", "character-female-a", "character-female-b"];

// ---------------- three.js setup ----------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2b2118);
scene.fog = new THREE.Fog(0x2b2118, 8, 20);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 2.4, 6.5);
camera.lookAt(0, 1.2, -0.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.getElementById('canvas-wrap').appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xfff3d6, 0x2a1d10, 1.1));
const sun = new THREE.DirectionalLight(0xffe9c4, 1.4);
sun.position.set(4, 8, 5);
sun.castShadow = true;
scene.add(sun);

// pavimento
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(30, 30),
  new THREE.MeshStandardMaterial({ color: 0x4a3826, roughness: 0.9 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// bancone del bar (primitive, nessun asset dedicato: e' l'unica geometria di scena
// costruita a mano, i personaggi/cibo sono asset veri Kenney CC0)
const counter = new THREE.Mesh(
  new THREE.BoxGeometry(4.4, 1.1, 0.9),
  new THREE.MeshStandardMaterial({ color: 0x6b4a2f, roughness: 0.6 })
);
counter.position.set(0, 0.55, 0.3);
counter.castShadow = true;
counter.receiveShadow = true;
scene.add(counter);

const shelf = new THREE.Mesh(
  new THREE.BoxGeometry(4.0, 0.12, 0.6),
  new THREE.MeshStandardMaterial({ color: 0x4a3320, roughness: 0.7 })
);
shelf.position.set(0, 1.55, -0.9);
shelf.castShadow = true;
scene.add(shelf);

// ---------------- la mosca (costruita a primitive) ----------------
const flyGroup = new THREE.Group();
const flyBodyMat = new THREE.MeshStandardMaterial({ color: 0x1c1c22, roughness: 0.3, metalness: 0.4 });
const body = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 16), flyBodyMat);
body.scale.set(1, 0.85, 1.3);
const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), flyBodyMat);
head.position.set(0, 0.05, 0.2);
const eyeMat = new THREE.MeshStandardMaterial({ color: 0xaa1122, roughness: 0.2 });
for (const s of [-1, 1]) {
  const eye = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 10), eyeMat);
  eye.position.set(s * 0.07, 0.06, 0.25);
  flyGroup.add(eye);
}
const wingMat = new THREE.MeshStandardMaterial({ color: 0xdfe9f5, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
const wings = [];
for (const s of [-1, 1]) {
  const pivot = new THREE.Group();
  pivot.position.set(s * 0.05, 0.08, 0);
  const wing = new THREE.Mesh(new THREE.PlaneGeometry(0.32, 0.14), wingMat);
  wing.position.set(s * 0.16, 0, 0);
  pivot.add(wing);
  flyGroup.add(pivot);
  wings.push(pivot);
}
flyGroup.add(body, head);
flyGroup.scale.setScalar(1.8);
scene.add(flyGroup);

const FLY_REST = new THREE.Vector3(0, 2.35, 0.15);
flyGroup.position.copy(FLY_REST);

// ---------------- caricamento asset GLB ----------------
const loader = new GLTFLoader();
function loadGLB(path) {
  return new Promise((resolve, reject) => loader.load(path, resolve, undefined, reject));
}

const foodSlots = {}; // item -> istanza sullo scaffale
const foodGltfCache = {}; // modelName -> gltf caricato, riusato per creare istanze sul bancone
const SHELF_X = { caffe: -1.4, cappuccino: -0.47, cornetto_semplice: 0.47, cornetto_nutella: 1.4 };
const COUNTER_X = SHELF_X; // stessa disposizione orizzontale, ma sul bancone invece che sullo scaffale

// i modelli del food-kit sono statici (non riggati), quindi .clone(true) qui e'
// sicuro - a differenza dei personaggi (vedi nota in spawnCustomer).
function createFoodInstance(item) {
  const modelName = FOOD_MODEL[item];
  const inst = foodGltfCache[modelName].scene.clone(true);
  inst.scale.setScalar(modelName === 'croissant' ? 2.2 : 2.0);
  inst.traverse(o => {
    if (o.isMesh) {
      o.castShadow = true;
      o.material = o.material.clone();
      if (item === 'cornetto_nutella') o.material.color.set(0x4a2c1a); // ponytail: stesso modello del cornetto semplice, tinto piu' scuro per suggerire nutella
    }
  });
  return inst;
}

async function setupFood() {
  for (const item of ITEMS) {
    const modelName = FOOD_MODEL[item];
    if (!foodGltfCache[modelName]) foodGltfCache[modelName] = await loadGLB(`assets/food/${modelName}.glb`);
    const inst = createFoodInstance(item);
    inst.position.set(SHELF_X[item], 1.55 + 0.12, -0.9);
    scene.add(inst);
    foodSlots[item] = inst;
  }
}

// ---- badge con la quantita' (sprite con testo disegnato su canvas) ----
function makeQuantityBadge(count) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffd27a';
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(4, 20, 120, 56, 16) : ctx.rect(4, 20, 120, 56);
  ctx.fill();
  ctx.strokeStyle = '#402c10'; ctx.lineWidth = 4; ctx.stroke();
  ctx.fillStyle = '#402c10';
  ctx.font = 'bold 48px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(`×${count}`, 64, 50);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(0.5, 0.375, 1);
  return sprite;
}

function makeNameTag(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(20,15,10,0.8)';
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(2, 8, 252, 48, 12) : ctx.rect(2, 8, 252, 48);
  ctx.fill();
  ctx.fillStyle = '#ffd27a';
  ctx.font = 'bold 32px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 34);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(1.0, 0.25, 1);
  sprite.userData.isNameTag = true;
  return sprite;
}

// prodotti appoggiati sul bancone dal cliente corrente, da rimuovere quando arriva il prossimo
let counterItems = [];
function clearCounter() {
  for (const obj of counterItems) scene.remove(obj);
  counterItems = [];
}

const characterCache = {};
async function getCharacter(name) {
  if (!characterCache[name]) characterCache[name] = await loadGLB(`assets/characters/${name}.glb`);
  return characterCache[name];
}

let currentCustomer = null; // { root, mixer, actions }
const CUSTOMER_IN = new THREE.Vector3(0, 0, 2.2);
const CUSTOMER_OUT = new THREE.Vector3(0, 0, 6.5);

async function spawnCustomer() {
  const name = CHARACTERS[Math.floor(Math.random() * CHARACTERS.length)];
  const gltf = await getCharacter(name);
  // ponytail: niente .clone(true) - su un modello skinned/riggato (questi lo sono,
  // hanno animazioni walk/idle/ecc.) il clone naive di Three.js non ricollega
  // correttamente lo scheletro e il personaggio risulta invisibile/deforme.
  // Mostriamo un solo cliente alla volta, quindi riusiamo l'oggetto originale
  // caricato invece di clonarlo (va aggiunto/rimosso dalla scena ogni volta).
  const root = gltf.scene;
  root.traverse(o => { if (o.isMesh) { o.castShadow = true; } });
  root.position.copy(CUSTOMER_OUT);
  root.rotation.y = Math.PI;
  // rimuove l'etichetta nome di un uso precedente di questo stesso modello
  // (root e' riusato, non clonato - vedi nota sopra)
  const oldTag = root.children.find(c => c.userData.isNameTag);
  if (oldTag) root.remove(oldTag);
  scene.add(root);
  const mixer = new THREE.AnimationMixer(root);
  const actions = {};
  for (const clip of gltf.animations) actions[clip.name] = mixer.clipAction(clip);
  currentCustomer = { root, mixer, actions, current: null };
  playAction('walk');
  return currentCustomer;
}

function playAction(name, opts = {}) {
  if (!currentCustomer || !currentCustomer.actions[name]) return;
  const { fade = 0.25, loop = THREE.LoopRepeat, once = false } = opts;
  const next = currentCustomer.actions[name];
  if (currentCustomer.current && currentCustomer.current !== next) currentCustomer.current.fadeOut(fade);
  next.reset().setLoop(once ? THREE.LoopOnce : loop, once ? 1 : Infinity);
  next.clampWhenFinished = once;
  next.fadeIn(fade).play();
  currentCustomer.current = next;
}

function despawnCustomer() {
  if (!currentCustomer) return;
  scene.remove(currentCustomer.root);
  currentCustomer = null;
}

// ---------------- animazione con semplici tween basati su tempo ----------------
function lerpVec(a, b, t) { return a.clone().lerp(b, t); }
function easeInOut(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

function animatePosition(obj, from, to, duration) {
  return new Promise(resolve => {
    const start = performance.now();
    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      obj.position.copy(lerpVec(from, to, easeInOut(t)));
      if (t < 1) requestAnimationFrame(step); else resolve();
    }
    requestAnimationFrame(step);
  });
}

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

async function flyVisitItem(item, count) {
  // 1. vola sullo scaffale e "raccoglie" (bounce sul modello originale)
  const shelfTarget = foodSlots[item].position.clone().add(new THREE.Vector3(0, 0.35, 0.15));
  await animatePosition(flyGroup, flyGroup.position.clone(), shelfTarget, 260);
  const mesh = foodSlots[item];
  const s0 = mesh.scale.clone();
  await animatePosition({ position: mesh.scale }, s0, s0.clone().multiplyScalar(1.35), 120);
  await animatePosition({ position: mesh.scale }, mesh.scale.clone(), s0, 160);

  // 2. porta una copia del prodotto sul bancone, con badge della quantita'
  const counterPos = new THREE.Vector3(COUNTER_X[item], 1.15, 0.35);
  await animatePosition(flyGroup, flyGroup.position.clone(), counterPos.clone().add(new THREE.Vector3(0, 0.3, 0)), 280);

  const placed = createFoodInstance(item);
  placed.position.copy(counterPos);
  placed.scale.setScalar(0.001); // pop-in invece di comparire di colpo
  scene.add(placed);
  counterItems.push(placed);

  const badge = makeQuantityBadge(count);
  badge.position.copy(counterPos).add(new THREE.Vector3(0, 0.5, 0));
  scene.add(badge);
  counterItems.push(badge);

  const fullScale = FOOD_MODEL[item] === 'croissant' ? 2.2 : 2.0;
  await animatePosition({ position: placed.scale }, placed.scale.clone(),
    new THREE.Vector3(fullScale, fullScale, fullScale), 180);
}

// ---------------- HUD / stato di gioco ----------------
const speechEl = document.getElementById('speech');
const bannerEl = document.getElementById('result-banner');
const btnServe = document.getElementById('btn-serve');
const btnAuto = document.getElementById('btn-auto');
const tagSelect = document.getElementById('tag-select');
const scoreEl = document.getElementById('score');
const totalEl = document.getElementById('total');
const queueBadge = document.getElementById('queue-badge');

let score = 0, total = 0, autoMode = false, busy = false, flyBusy = false;

function showBanner(text, color) {
  bannerEl.textContent = text;
  bannerEl.style.color = color;
  bannerEl.style.display = 'block';
  setTimeout(() => bannerEl.style.display = 'none', 1100);
}

const modelBadgeEl = document.getElementById('model-badge');
const modelInfoEl = document.getElementById('model-info');
const probsViewEl = document.getElementById('probs-view');
let modelInfoCache = null;

async function loadModelInfo() {
  const r = await fetch('/api/model_info');
  modelInfoCache = await r.json();
  updateModelBadge();
}

function updateModelBadge() {
  if (!modelInfoCache) return;
  const info = modelInfoCache[tagSelect.value];
  if (!info) return;
  if (info.trained) {
    const kb = (info.checkpoint_bytes / 1024).toFixed(0);
    const date = new Date(info.checkpoint_mtime * 1000).toLocaleString();
    modelBadgeEl.innerHTML = `✅ allenata — ${info.n_params.toLocaleString()} parametri<br>checkpoint: ${info.checkpoint_file} (${kb} KB, salvato ${date})`;
  } else {
    modelBadgeEl.innerHTML = `⚠️ <b>MAI allenata</b> — pesi random, ${info.n_params.toLocaleString()} parametri, nessun file checkpoint`;
  }
}
tagSelect.addEventListener('change', updateModelBadge);

function renderProbs(data) {
  if (!data.probs) return;
  let html = `<div style="margin-bottom:4px;">rete: <b>${data.tag}</b> ${data.model_trained ? '(allenata)' : '(⚠️ MAI allenata)'}</div>`;
  ITEMS.forEach((item, i) => {
    const probs = data.probs[i]; // [P(0), P(1), P(2), P(3)]
    const chosen = data.served[i];
    const bars = probs.map((p, k) => {
      const pct = Math.round(p * 100);
      const hl = k === chosen ? 'background:#ffd27a;' : 'background:#5a4a38;';
      return `<div style="display:inline-block;width:22px;text-align:center;margin-right:2px;">
        <div style="height:${Math.max(2, pct * 0.4)}px; ${hl} border-radius:2px;"></div>
        <div style="font-size:9px;">${k}</div></div>`;
    }).join('');
    html += `<div style="margin-top:4px;">${ITEM_LABELS[item]} (vero: ${data.order[i]})<br>
      <div style="display:flex; align-items:flex-end; height:40px;">${bars}</div></div>`;
  });
  probsViewEl.innerHTML = html;
}

const queueListEl = document.getElementById('queue-list');
async function refreshQueueBadge() {
  try {
    const r = await fetch('/api/queue');
    const data = await r.json();
    queueBadge.textContent = `${data.length} in coda`;
    queueListEl.innerHTML = data.orders.length
      ? data.orders.map(o => `👤 <b>${o.nickname}</b>: ${o.text}`).join('<br>')
      : '<span style="opacity:0.6">nessun cliente vero in coda</span>';
  } catch (e) { /* server non ancora su, ignora */ }
}

async function doServe() {
  if (busy) return;
  busy = true;
  btnServe.disabled = true;
  clearCounter();

  const cust = await spawnCustomer();
  await animatePosition(cust.root, CUSTOMER_OUT, CUSTOMER_IN, 900);
  playAction('idle');

  const res = await fetch(`/api/next_customer?tag=${encodeURIComponent(tagSelect.value)}`);
  const data = await res.json();

  speechEl.textContent = `${data.nickname}: "${data.order_text}"`;
  speechEl.style.display = 'block';
  const nameTag = makeNameTag(data.nickname);
  nameTag.position.set(0, 1.9, 0);
  cust.root.add(nameTag);
  renderProbs(data);
  await wait(500);

  flyBusy = true;
  for (const item of ITEMS) {
    const count = data.served[ITEMS.indexOf(item)];
    if (count > 0) await flyVisitItem(item, count);
  }
  await animatePosition(flyGroup, flyGroup.position.clone(), FLY_REST, 300);
  flyBusy = false;

  total += 1;
  if (data.all_correct) {
    score += 1;
    playAction('emote-yes', { once: true, fade: 0.1 });
    showBanner('GIUSTO! +1 🍯', '#7CFC7C');
  } else {
    playAction('emote-no', { once: true, fade: 0.1 });
    showBanner('SBAGLIATO -1', '#ff6b6b');
  }
  scoreEl.textContent = score;
  totalEl.textContent = total;

  await wait(1000);
  speechEl.style.display = 'none';
  playAction('walk');
  await animatePosition(cust.root, CUSTOMER_IN, CUSTOMER_OUT, 700);
  despawnCustomer();
  await refreshQueueBadge();

  busy = false;
  btnServe.disabled = false;
  if (autoMode) doServe();
}

btnServe.addEventListener('click', doServe);
btnAuto.addEventListener('click', () => {
  autoMode = !autoMode;
  btnAuto.textContent = autoMode ? '⏸️ Stop auto' : '▶️ Auto';
  if (autoMode && !busy) doServe();
});

// ---------------- form ordine cliente esterno ----------------
const qtyRowsEl = document.getElementById('qty-rows');
const qtyInputs = {};
for (const item of ITEMS) {
  const row = document.createElement('div');
  row.className = 'qty-row';
  row.innerHTML = `<label>${ITEM_LABELS[item]}</label>`;
  const input = document.createElement('input');
  input.type = 'number'; input.min = 0; input.max = MAX_COUNT; input.value = 0; input.style.width = '50px';
  qtyInputs[item] = input;
  row.appendChild(input);
  qtyRowsEl.appendChild(row);
}
document.getElementById('btn-order').addEventListener('click', async () => {
  const nickname = document.getElementById('nickname').value.trim();
  const items = ITEMS.map(it => parseInt(qtyInputs[it].value || '0', 10));
  const msgEl = document.getElementById('order-msg');
  if (!nickname) { msgEl.textContent = 'Metti un nickname.'; return; }
  if (items.every(v => v === 0)) { msgEl.textContent = 'Ordina almeno qualcosa!'; return; }
  const res = await fetch('/api/order', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nickname, items })
  });
  const data = await res.json();
  msgEl.textContent = data.ok ? `In coda! (${data.queue_length} in attesa)` : data.error;
  await refreshQueueBadge();
});

// ---------------- loop di rendering ----------------
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  const t = clock.elapsedTime;

  if (!flyBusy) flyGroup.position.y = FLY_REST.y + Math.sin(t * 3) * 0.05; // bob solo quando non sta volando verso un prodotto
  for (const w of wings) w.rotation.z = Math.sin(t * 40) * 0.9;

  if (currentCustomer) currentCustomer.mixer.update(dt);
  renderer.render(scene, camera);
}

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

setupFood().then(() => { refreshQueueBadge(); animate(); });
loadModelInfo();
setInterval(refreshQueueBadge, 4000);
