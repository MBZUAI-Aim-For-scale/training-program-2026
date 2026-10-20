// ==========================================================================
// AI Weather — Minimal Vanilla JS for Interactivity
// ==========================================================================

(function() {
  'use strict';

  // --- Theme Toggle ---
  function initTheme() {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? 'dark' : 'light');
    applyTheme(theme);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
  }

  function updateThemeIcon(theme) {
    const sun = document.querySelector('.sun-icon');
    const moon = document.querySelector('.moon-icon');
    if (sun && moon) {
      sun.style.display = theme === 'dark' ? 'none' : 'block';
      moon.style.display = theme === 'dark' ? 'block' : 'none';
    }
  }

  window.toggleTheme = function() {
    const current = document.documentElement.getAttribute('data-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const newTheme = (current || (prefersDark ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  };

  // --- Mobile Menu ---
  window.toggleMobileMenu = function() {
    const nav = document.getElementById('mobile-nav');
    if (nav) nav.classList.toggle('open');
  };

  // --- Table of Contents Generation ---
  function buildTOC() {
    const tocContainer = document.getElementById('toc');
    if (!tocContainer) return;

    const headings = document.querySelectorAll('.episode-content h2, .episode-content h3');
    if (headings.length === 0) {
      tocContainer.parentElement.style.display = 'none';
      return;
    }

    const list = document.createElement('ul');
    list.style.listStyle = 'none';
    list.style.padding = '0';
    list.style.margin = '0';

    headings.forEach((heading, index) => {
      // Add anchor link to heading
      if (!heading.id) {
        heading.id = 'section-' + index;
      }
      
      const anchor = document.createElement('a');
      anchor.href = '#' + heading.id;
      anchor.textContent = heading.textContent;
      anchor.className = heading.tagName === 'H3' ? 'toc-h3' : 'toc-h2';
      
      const item = document.createElement('li');
      item.appendChild(anchor);
      list.appendChild(item);
    });

    tocContainer.appendChild(list);

    // Active section tracking
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const id = entry.target.id;
        const link = tocContainer.querySelector(`a[href="#${id}"]`);
        if (link) {
          if (entry.isIntersecting) {
            tocContainer.querySelectorAll('a').forEach(a => a.classList.remove('active'));
            link.classList.add('active');
          }
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });

    headings.forEach(h => observer.observe(h));
  }

  // --- Add anchor links to headings ---
  function addHeadingAnchors() {
    document.querySelectorAll('.episode-content h1, .episode-content h2, .episode-content h3, .episode-content h4').forEach(heading => {
      if (!heading.id) {
        heading.id = heading.textContent.toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .trim();
      }
      
      const anchor = document.createElement('a');
      anchor.href = '#' + heading.id;
      anchor.className = 'anchor';
      anchor.setAttribute('aria-label', 'Link to ' + heading.textContent);
      anchor.textContent = '#';
      heading.appendChild(anchor);
    });
  }

  // --- Copy buttons for code blocks ---
  function addCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.setAttribute('aria-label', 'Copy code');
      btn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
        <span>Copy</span>
      `;
      
      btn.addEventListener('click', async () => {
        const code = pre.querySelector('code');
        if (!code) return;
        
        try {
          await navigator.clipboard.writeText(code.textContent);
          btn.querySelector('span').textContent = 'Copied!';
          setTimeout(() => {
            btn.querySelector('span').textContent = 'Copy';
          }, 2000);
        } catch (err) {
          console.error('Copy failed:', err);
        }
      });
      
      pre.style.position = 'relative';
      pre.appendChild(btn);
    });
  }

  // --- Keyboard shortcuts ---
  document.addEventListener('keydown', (e) => {
    // Cmd/Ctrl + K for search
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      if (typeof openSearch === 'function') openSearch();
    }
    
    // 'n' for next, 'p' for previous (only when not in input)
    if (document.activeElement.tagName !== 'INPUT' && 
        document.activeElement.tagName !== 'TEXTAREA') {
      const nextLink = document.querySelector('.pagination-link.next');
      const prevLink = document.querySelector('.pagination-link.prev');
      
      if (e.key === 'n' && nextLink) {
        window.location.href = nextLink.href;
      } else if (e.key === 'p' && prevLink) {
        window.location.href = prevLink.href;
      }
    }
  });

  // --- Search placeholder ---
  window.openSearch = function() {
    alert('Search coming soon! (Integrate with Pagefind or Lunr.js)');
  };

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, null, this.getAttribute('href'));
      }
    });
  });

  // --- Initialize ---
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    addHeadingAnchors();
    buildTOC();
    addCopyButtons();
  });

  // Run immediately if DOM already loaded
  if (document.readyState !== 'loading') {
    initTheme();
    addHeadingAnchors();
    buildTOC();
    addCopyButtons();
  }
})();