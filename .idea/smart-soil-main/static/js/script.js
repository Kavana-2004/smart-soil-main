// Navbar active link highlighting
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.navbar ul li a');

    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (scrollY >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(current)) {
            link.classList.add('active');
        }
    });
});

// Prevent page jump to top when form is submitted
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Store current scroll position
            sessionStorage.setItem('scrollPosition', window.scrollY);
        });
    }

    // Restore scroll position after form submission (if results are shown)
    const scrollPosition = sessionStorage.getItem('scrollPosition');
    if (scrollPosition && document.querySelector('.prediction-result')) {
        setTimeout(() => {
            window.scrollTo(0, parseInt(scrollPosition));
            sessionStorage.removeItem('scrollPosition');
        }, 100);
    }

    // Alternative: Scroll to results smoothly when they appear
    if (document.querySelector('.prediction-result')) {
        setTimeout(() => {
            const resultsSection = document.getElementById('results');
            if (resultsSection) {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 500);
    }
});

// Optional: If you want to keep the slider (though you're using carousel now)
let slides = document.querySelectorAll(".slider .slide");
if (slides.length > 0) {
    let currentSlide = 0;

    function showNextSlide() {
        slides[currentSlide].classList.remove("active");
        currentSlide = (currentSlide + 1) % slides.length;
        slides[currentSlide].classList.add("active");
    }

    setInterval(showNextSlide, 3000);
}