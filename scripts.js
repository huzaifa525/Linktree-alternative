/* ========================================
   FAST 2025 AI ENGINEER PORTFOLIO
   Three.js + GSAP ScrollTrigger + WebGL
   ======================================== */

// ============================================
// CONFIG
// ============================================
const CONFIG = {
    githubUsername: 'huzefanalkheda',
    enable3D: true,
    enableGSAP: true
};

// ============================================
// 1. THREE.JS 3D BACKGROUND WITH PARTICLES
// ============================================
class ThreeBackground {
    constructor() {
        this.canvas = document.getElementById('webgl-canvas');
        if (!this.canvas) return;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: true,
            antialias: true
        });

        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.camera.position.z = 5;

        this.createParticles();
        this.createLights();
        this.setupEventListeners();
        this.animate();
    }

    createParticles() {
        const geometry = new THREE.BufferGeometry();
        const particleCount = 2000;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        const colorPalette = [
            new THREE.Color(0x00d4ff), // neon-blue
            new THREE.Color(0xb026ff), // neon-purple
            new THREE.Color(0xff006e), // neon-pink
            new THREE.Color(0x00ff88)  // neon-green
        ];

        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 20;
            positions[i + 1] = (Math.random() - 0.5) * 20;
            positions[i + 2] = (Math.random() - 0.5) * 10;

            const color = colorPalette[Math.floor(Math.random() * colorPalette.length)];
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.8
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    createLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0x00d4ff, 1, 100);
        pointLight.position.set(0, 0, 10);
        this.scene.add(pointLight);
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onResize());
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    onMouseMove(event) {
        const x = (event.clientX / window.innerWidth) * 2 - 1;
        const y = -(event.clientY / window.innerHeight) * 2 + 1;

        gsap.to(this.camera.position, {
            x: x * 0.5,
            y: y * 0.5,
            duration: 1,
            ease: 'power2.out'
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Rotate particles slowly
        this.particles.rotation.y += 0.0005;
        this.particles.rotation.x += 0.0002;

        this.renderer.render(this.scene, this.camera);
    }
}

// ============================================
// 2. GSAP SCROLLTRIGGER ANIMATIONS
// ============================================
function initGSAPAnimations() {
    gsap.registerPlugin(ScrollTrigger);

    // Simple fade in for hero - no excessive animation
    gsap.set('.hero-text h1, .subtitle, .description, .hero-stats, .cta-buttons', { opacity: 0 });

    gsap.to('.hero-text h1, .subtitle, .description, .hero-stats, .cta-buttons', {
        opacity: 1,
        duration: 0.8,
        ease: 'power2.out',
        stagger: 0.1
    });

    // Section titles - trigger earlier
    gsap.utils.toArray('.section-title').forEach(title => {
        gsap.from(title, {
            scrollTrigger: {
                trigger: title,
                start: 'top 95%',  // Trigger sooner
            },
            opacity: 0,
            y: 30,
            duration: 0.6,
            ease: 'power2.out'
        });
    });

    // Project cards - trigger earlier
    gsap.utils.toArray('.project-card').forEach(card => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 90%',  // Trigger sooner
            },
            opacity: 0,
            y: 40,
            duration: 0.6,
            ease: 'power2.out'
        });
    });

    // Skill categories - trigger earlier
    gsap.utils.toArray('.skill-category').forEach((skill, index) => {
        gsap.from(skill, {
            scrollTrigger: {
                trigger: skill,
                start: 'top 90%',  // Trigger sooner
            },
            opacity: 0,
            y: 30,
            duration: 0.6,
            ease: 'power2.out',
            delay: index * 0.05
        });
    });

    // Parallax effect for hero image
    gsap.to('.hero-image', {
        scrollTrigger: {
            trigger: '.hero',
            start: 'top top',
            end: 'bottom top',
            scrub: 1
        },
        y: 200,
        ease: 'none'
    });
}

// ============================================
// 3. GLASS NAVBAR
// ============================================
function initGlassNav() {
    const nav = document.querySelector('.glass-nav');
    const mobileToggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.glass-nav-links');

    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.top = '10px';
        } else {
            nav.style.top = '20px';
        }
    });

    // Mobile toggle
    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            const isExpanded = navLinks.classList.contains('active');
            mobileToggle.setAttribute('aria-expanded', isExpanded);

            const icon = mobileToggle.querySelector('i');
            if (icon) {
                icon.classList.toggle('fa-bars');
                icon.classList.toggle('fa-times');
            }
        });

        // Close on link click
        document.querySelectorAll('.glass-nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                mobileToggle.setAttribute('aria-expanded', 'false');
            });
        });
    }
}

// ============================================
// 4. CUSTOM CURSOR
// ============================================
function initCustomCursor() {
    if ('ontouchstart' in window) return;

    const cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    document.body.appendChild(cursor);

    let mouseX = 0, mouseY = 0;
    let cursorX = 0, cursorY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function animateCursor() {
        cursorX += (mouseX - cursorX) * 0.1;
        cursorY += (mouseY - cursorY) * 0.1;
        cursor.style.transform = `translate(${cursorX - 15}px, ${cursorY - 15}px)`;
        requestAnimationFrame(animateCursor);
    }
    animateCursor();

    const interactiveElements = document.querySelectorAll('a, button, .project-card, .stat');
    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => cursor.classList.add('hover'));
        el.addEventListener('mouseleave', () => cursor.classList.remove('hover'));
    });
}

// ============================================
// 5. SCROLL PROGRESS BAR
// ============================================
function initScrollProgress() {
    const progressBar = document.querySelector('.scroll-progress');
    if (!progressBar) return;

    window.addEventListener('scroll', () => {
        const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrolled = (window.scrollY / scrollHeight) * 100;
        progressBar.style.width = `${scrolled}%`;
    });
}

// ============================================
// 6. STATS COUNTER ANIMATION
// ============================================
function initStatsCounter() {
    const stats = document.querySelectorAll('.stat-number, .github-stat-number');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const text = target.textContent.trim();

                if (!/\d/.test(text)) return;

                const match = text.match(/([≈~])?(\d+)([K\+]*)/i);
                if (!match) return;

                const prefix = match[1] || '';
                const number = parseInt(match[2]);
                const suffix = match[3] || '';

                // Skip animation for single-digit numbers
                if (number < 10) {
                    target.textContent = prefix + number + suffix;
                    observer.unobserve(target);
                    return;
                }

                let current = 0;
                const increment = number / 50;
                const timer = setInterval(() => {
                    current += increment;
                    if (current >= number) {
                        current = number;
                        clearInterval(timer);
                    }
                    target.textContent = prefix + Math.floor(current) + suffix;
                }, 30);

                observer.unobserve(target);
            }
        });
    }, { threshold: 0.5 });

    stats.forEach(stat => observer.observe(stat));
}

// ============================================
// 7. 3D CARD TILT EFFECT
// ============================================
function initProjectCardTilt() {
    const cards = document.querySelectorAll('.project-card, .stat, .skill-category');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 15;
            const rotateY = (centerX - x) / 15;

            gsap.to(card, {
                rotationX: rotateX,
                rotationY: rotateY,
                transformPerspective: 1000,
                duration: 0.5,
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                rotationX: 0,
                rotationY: 0,
                duration: 0.5,
                ease: 'power2.out'
            });
        });
    });
}

// ============================================
// 8. GITHUB CONTRIBUTION GRAPH
// ============================================
async function initGitHubGraph() {
    const container = document.getElementById('github-graph');
    if (!container) return;

    try {
        const response = await fetch(`https://github-contributions-api.jogruber.de/v4/${CONFIG.githubUsername}?y=2025`);
        if (!response.ok) throw new Error('Failed to fetch');

        const data = await response.json();
        createContributionGraph(container, data.contributions);
    } catch (error) {
        console.log('GitHub graph unavailable');
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <a href="https://github.com/${CONFIG.githubUsername}"
                   target="_blank"
                   style="color: var(--neon-blue); text-decoration: none; font-weight: 700;">
                    <i class="fab fa-github"></i> View GitHub Profile
                </a>
            </div>
        `;
    }
}

function createContributionGraph(container, contributions) {
    const weeks = Object.values(contributions);
    const graph = document.createElement('div');
    graph.style.cssText = 'display: grid; grid-auto-flow: column; gap: 3px; justify-content: center;';

    weeks.forEach(week => {
        const weekColumn = document.createElement('div');
        weekColumn.style.cssText = 'display: grid; gap: 3px;';

        week.forEach(day => {
            const cell = document.createElement('div');
            const level = day.count === 0 ? 0 : Math.min(Math.ceil(day.count / 5), 4);
            const colors = ['rgba(0, 212, 255, 0.1)', 'rgba(0, 212, 255, 0.3)', 'rgba(0, 212, 255, 0.6)', 'rgba(0, 212, 255, 0.8)', 'rgba(0, 212, 255, 1)'];

            cell.style.cssText = `
                width: 12px;
                height: 12px;
                background: ${colors[level]};
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 2px;
                transition: all 0.3s;
                cursor: pointer;
            `;
            cell.title = `${day.date}: ${day.count} contributions`;

            cell.addEventListener('mouseenter', () => {
                gsap.to(cell, { scale: 1.5, boxShadow: '0 0 10px var(--neon-blue)', duration: 0.2 });
            });
            cell.addEventListener('mouseleave', () => {
                gsap.to(cell, { scale: 1, boxShadow: 'none', duration: 0.2 });
            });

            weekColumn.appendChild(cell);
        });

        graph.appendChild(weekColumn);
    });

    container.innerHTML = '';
    container.appendChild(graph);
}

// ============================================
// 9. SMOOTH SCROLL
// ============================================
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#' || !href) return; // Skip empty or just # links

            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navHeight = 100; // Navbar height offset
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// ============================================
// 10. LAZY LOADING
// ============================================
function initLazyLoading() {
    const images = document.querySelectorAll('img[loading="lazy"]');

    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src || img.src;
                img.classList.add('loaded');
                imageObserver.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
}

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 FAST 2025 Portfolio Initializing...');

    // Core functionality
    initGlassNav();
    initCustomCursor();
    initScrollProgress();
    initSmoothScroll();

    // Three.js 3D Background
    if (CONFIG.enable3D) {
        new ThreeBackground();
        console.log('✨ Three.js 3D Active');
    }

    // GSAP ScrollTrigger
    if (CONFIG.enableGSAP) {
        initGSAPAnimations();
        console.log('✨ GSAP ScrollTrigger Active');
    }

    // Animations
    initStatsCounter();
    initProjectCardTilt();

    // Data
    // GitHub graph now uses direct image embed - no JS needed
    // initGitHubGraph();

    // Lazy loading
    initLazyLoading();

    console.log('✅ Portfolio Ready!');
});

// ============================================
// PERFORMANCE MONITORING
// ============================================
if (performance && performance.mark) {
    window.addEventListener('load', () => {
        performance.mark('portfolio-loaded');
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log(`⚡ Load Time: ${Math.round(perfData.loadEventEnd - perfData.fetchStart)}ms`);
    });
}
