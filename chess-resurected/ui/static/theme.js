// Theme bootstrap. Default is LIGHT (creamy). Toggle persists in localStorage.
(function(){
  var KEY = 'cr-theme';
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch(e){}
  var theme = saved || 'light';
  document.documentElement.setAttribute('data-theme', theme);

  function applyAndPersist(t){
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(KEY, t); } catch(e){}
  }

  document.addEventListener('DOMContentLoaded', function(){
    var btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.setAttribute('aria-label', 'Toggle light/dark mode');
    btn.title = 'Toggle theme';
    btn.innerHTML = '<span class="ico"></span>';
    btn.addEventListener('click', function(){
      var cur = document.documentElement.getAttribute('data-theme') || 'light';
      applyAndPersist(cur === 'light' ? 'dark' : 'light');
    });
    document.body.appendChild(btn);
  });
})();
