// custom.js
document.addEventListener('DOMContentLoaded', function() {
    // Add logo to the date in the title slide
    const titleSlide = document.querySelector('.title-slide');
    if (titleSlide) {
        const dateElement = titleSlide.querySelector('p.date');
        if (dateElement) {
            const logo = document.createElement('img');
            logo.src = 'fig/ephi_logo.png';
            logo.style.height = '40px';
            logo.style.verticalAlign = 'middle';
            logo.style.marginRight = '10px';
            dateElement.insertBefore(logo, dateElement.firstChild);
        }
    }
    
    // Optionally add to footer
    const footer = document.querySelector('.reveal .footer');
    if (footer) {
        const logo = document.createElement('img');
        logo.src = 'fig/ephi_logo.png';
        logo.style.height = '25px';
        logo.style.verticalAlign = 'middle';
        logo.style.marginRight = '8px';
        footer.insertBefore(logo, footer.firstChild);
    }
});