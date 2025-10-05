

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 100000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

const earthGeometry = new THREE.SphereGeometry( 6371, 32, 32 );
const earthMaterial = new THREE.MeshPhongMaterial({
    map: new THREE.TextureLoader().load('https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg'),
    bumpMap: new THREE.TextureLoader().load('https://www.solarsystemscope.com/textures/download/8k_earth_normal_map.jpg'),
    bumpScale: 1,
    specularMap: new THREE.TextureLoader().load('https://www.solarsystemscope.com/textures/download/8k_earth_specular_map.jpg'),
    specular: new THREE.Color('grey')
});
const earth = new THREE.Mesh( earthGeometry, earthMaterial );
scene.add( earth );

const ambientLight = new THREE.AmbientLight( 0x333333 );
scene.add( ambientLight );

const directionalLight = new THREE.DirectionalLight( 0xffffff, 1 );
directionalLight.position.set( 5, 3, 5 );
scene.add( directionalLight );

const satellites = [];
const orbits = [];

fetch('leo_satellites.txt')
    .then(response => response.text())
    .then(data => {
        const tleLines = data.trim().split('\n').slice(0, 200); // Process only the first 100 satellites
        for (let i = 0; i < tleLines.length; i += 2) {
            const tleLine1 = tleLines[i].trim();
            const tleLine2 = tleLines[i + 1].trim();

            if (!tleLine1 || !tleLine2) continue;

            const satrec = satellite.twoline2satrec(tleLine1, tleLine2);

            const satGeometry = new THREE.BoxGeometry( 50, 50, 50 );
            const satMaterial = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
            const satMesh = new THREE.Mesh( satGeometry, satMaterial );
            satellites.push({ mesh: satMesh, satrec: satrec });
            scene.add( satMesh );

            const orbitPoints = [];
            const now = new Date();
            for (let j = 0; j < 360; j++) {
                const time = new Date(now.getTime() + j * 60000);
                const positionAndVelocity = satellite.propagate(satrec, time);
                const positionEci = positionAndVelocity.position;
                if (positionEci) {
                    orbitPoints.push(new THREE.Vector3(positionEci.x, positionEci.y, positionEci.z));
                }
            }
            const orbitGeometry = new THREE.BufferGeometry().setFromPoints( orbitPoints );
            const orbitMaterial = new THREE.LineBasicMaterial( { color: 0x0000ff } );
            const orbit = new THREE.Line( orbitGeometry, orbitMaterial );
            orbits.push(orbit);
            scene.add( orbit );
        }
    });

camera.position.z = 15000;

function animate() {
    requestAnimationFrame( animate );

    const now = new Date();
    for (const sat of satellites) {
        try {
            const positionAndVelocity = satellite.propagate(sat.satrec, now);
            const positionEci = positionAndVelocity.position;
            if (positionEci) {
                sat.mesh.position.x = positionEci.x;
                sat.mesh.position.y = positionEci.y;
                sat.mesh.position.z = positionEci.z;
            }
        } catch (e) {
            // Ignore propagation errors for now
        }
    }

    earth.rotation.y += 0.0005;

    renderer.render( scene, camera );
}

animate();
