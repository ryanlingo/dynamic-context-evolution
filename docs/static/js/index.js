// Minimal interactivity for the project page
window.addEventListener('DOMContentLoaded', () => {
  // Navbar burger toggle for mobile
  const burgers = document.querySelectorAll('.navbar-burger');
  burgers.forEach(burger => {
    burger.addEventListener('click', () => {
      const target = document.getElementById(burger.dataset.target);
      burger.classList.toggle('is-active');
      target.classList.toggle('is-active');
    });
  });
});
