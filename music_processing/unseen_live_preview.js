// Live 3D preview for the demo's Edit tab: the run's model and dance in the browser, filmed by a camera VMD
// that is swapped every time a label changes. The final MP4 still comes from the SABA renderer.
import * as THREE from "three";
import { MMDLoader } from "three/addons/loaders/MMDLoader.js";
import { MMDAnimationHelper } from "three/addons/animation/MMDAnimationHelper.js";

const AMMO_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/ammo.wasm.js";
const ASPECT = 16 / 9;
let ammoPromise = null;

function loadAmmo() {
  if (!ammoPromise) {
    ammoPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = AMMO_URL;
      script.onload = () => window.Ammo().then((lib) => {
        window.Ammo = lib;
        resolve(lib);
      }, reject);
      script.onerror = () => reject(new Error("Could not load the physics library."));
      document.head.appendChild(script);
    });
  }
  return ammoPromise;
}

function backgroundGradient() {
  // Dark at the top, lighter toward the floor, like the SABA preview background.
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0, "#343434");
  gradient.addColorStop(1, "#4c4c4c");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// three.js reads a VMD's bone, morph, and camera frames but skips its IK on/off frames. Dances that switch
// leg IK off and pose the legs by FK need them, or the feet get pulled toward the unused IK bones.
async function loadIkSwitches(url) {
  const buffer = await (await fetch(url)).arrayBuffer();
  const view = new DataView(buffer);
  let offset = 50;
  for (const recordSize of [111, 23, 61, 28, 9]) {
    if (offset + 4 > view.byteLength) return [];
    offset += 4 + view.getUint32(offset, true) * recordSize;
  }
  if (offset + 4 > view.byteLength) return [];
  const decoder = new TextDecoder("shift_jis");
  const switches = [];
  for (let i = view.getUint32(offset, true), at = offset + 4; i > 0; i -= 1) {
    const frame = view.getUint32(at, true);
    const count = view.getUint32(at + 5, true);
    at += 9;
    const off = [];
    for (let k = 0; k < count; k += 1, at += 21) {
      const bytes = new Uint8Array(buffer, at, 20);
      const end = bytes.indexOf(0);
      if (view.getUint8(at + 20) === 0) off.push(decoder.decode(bytes.subarray(0, end < 0 ? 20 : end)));
    }
    switches.push({ frame, off });
  }
  return switches.sort((a, b) => a.frame - b.frame);
}

function holdLastFrame(mixer) {
  for (const action of mixer._actions) {
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
  }
}

export class LivePreview {
  constructor(container, clock) {
    this.container = container;
    this.clock = clock;
    // Textures arrive after the mesh; draw again as each one lands.
    const manager = new THREE.LoadingManager();
    manager.onProgress = () => { this.dirty = true; };
    this.loader = new MMDLoader(manager);
    this.helper = null;
    this.mesh = null;
    this.cameraUrl = "";
    this.lastTime = -1;
    this.dirty = true;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = backgroundGradient();
    // Matched to the SABA preview by comparing model pixel colors at the same frames.
    this.scene.add(new THREE.AmbientLight(0xaaaaaa, 1.0));
    const light = new THREE.DirectionalLight(0xffffff, 0.6 * Math.PI);
    light.position.set(-20, 40, 20);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    Object.assign(light.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 150 });
    light.shadow.camera.updateProjectionMatrix();
    this.scene.add(light);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.ShadowMaterial({ opacity: 0.28 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    this.camera = new THREE.PerspectiveCamera(30, ASPECT, 0.5, 2000);
    this.camera.position.set(0, 10, 40);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.frame = requestAnimationFrame(() => this.loop());
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    const w = Math.min(width, height * ASPECT);
    this.renderer.setSize(Math.round(w), Math.round(w / ASPECT));
    this.dirty = true;
  }

  async load({ modelUrl, motionUrl, cameraUrl, onStatus = () => {} }) {
    let physics = true;
    onStatus("Loading the physics engine...");
    await loadAmmo().catch(() => { physics = false; });
    onStatus("Loading the model and the dance...");
    const { mesh, animation } = await new Promise((resolve, reject) => {
      this.loader.loadWithAnimation(modelUrl, motionUrl, resolve, undefined, reject);
    });
    mesh.castShadow = true;
    if (this.helper) {
      // The old helper's camera target stays a child of the camera unless removed, and the new
      // camera clip would then animate that stale target while the camera looks at the new one.
      if (this.helper.camera) this.helper.remove(this.camera);
      this.helper.remove(this.mesh);
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      for (const material of [this.mesh.material].flat()) {
        material.map?.dispose();
        material.dispose();
      }
    }
    this.mesh = mesh;
    this.scene.add(mesh);
    this.helper = new MMDAnimationHelper();
    this.helper.add(mesh, { animation, physics });
    holdLastFrame(this.helper.objects.get(mesh).mixer);
    this.watchIkSwitches(mesh, await loadIkSwitches(motionUrl).catch(() => []));
    await this.setCamera(cameraUrl);
    onStatus("");
  }

  async setCamera(url) {
    if (!url || !this.helper) return;
    const clip = await new Promise((resolve, reject) => this.loader.loadAnimation(url, this.camera, resolve, undefined, reject));
    if (this.helper.camera) this.helper.remove(this.camera);
    this.helper.add(this.camera, { animation: clip });
    holdLastFrame(this.helper.objects.get(this.camera).mixer);
    this.cameraUrl = url;
    this.seek(this.clock());
  }

  watchIkSwitches(mesh, switches) {
    const boneIndex = new Map(mesh.skeleton.bones.map((bone, index) => [bone.name, index]));
    this.ikSwitches = switches.map(({ frame, off }) => ({ frame, off: new Set(off.map((name) => boneIndex.get(name))) }));
    this.ikOff = new Set();
    const solver = this.helper.objects.get(mesh)?.ikSolver;
    if (!solver || !this.ikSwitches.length) return;
    // Both of the helper's IK paths (PMX bone order and plain update) solve one IK at a time through updateOne.
    const solve = solver.updateOne.bind(solver);
    solver.updateOne = (ik) => (this.ikOff.has(ik.target) ? solver : solve(ik));
  }

  applyIkSwitches(sec) {
    let current = null;
    for (const entry of this.ikSwitches || []) {
      if (entry.frame > sec * 30 + 1e-3) break;
      current = entry;
    }
    this.ikOff = current ? current.off : new Set();
  }

  seek(sec) {
    if (!this.helper) return;
    // The helper rewinds the bones to its saved animation pose before every update, and the mixer only writes
    // values that changed. Rewind first, jump, then save the new pose, or a paused jump keeps the old pose.
    this.helper._restoreBones(this.mesh);
    for (const object of [this.mesh, this.camera]) {
      const mixer = this.helper.objects.get(object)?.mixer;
      if (!mixer) continue;
      for (const action of mixer._actions) action.reset().play();
      mixer.setTime(sec);
    }
    this.helper._saveBones(this.mesh);
    this.applyIkSwitches(sec);
    this.helper.update(0);
    const physics = this.helper.objects.get(this.mesh)?.physics;
    if (physics) {
      // Let hair and clothes settle on the new pose instead of snapping from the old one.
      physics.reset();
      for (let i = 0; i < 30; i += 1) physics.update(1 / 60);
    }
    this.lastTime = sec;
    this.dirty = true;
  }

  loop() {
    this.frame = requestAnimationFrame(() => this.loop());
    if (this.helper) {
      const sec = this.clock();
      const delta = sec - this.lastTime;
      if (delta > 0 && delta < 0.25) {
        this.applyIkSwitches(sec);
        this.helper.update(delta);
        this.lastTime = sec;
        this.dirty = true;
      } else if (Math.abs(delta) > 1e-4) {
        this.seek(sec);
      }
    }
    if (this.dirty) {
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
  }
}
