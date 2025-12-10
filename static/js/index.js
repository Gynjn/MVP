window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("teaser");
  const loader = document.getElementById("teaser-loader");

  if (video && loader) {
    
    const hideLoader = () => {
      // 1. Fade it out
      loader.classList.add("is-hidden");
      // 2. Stop checking (save resources)
      video.removeEventListener("canplay", hideLoader);
      video.removeEventListener("loadeddata", hideLoader);
    };

    // CASE A: The video is already loaded in the cache
    if (video.readyState >= 3) {
      hideLoader();
    } 
    // CASE B: The video is still downloading
    else {
      // 'canplay' means the browser has enough data to start
      video.addEventListener("canplay", hideLoader);
      // Fallback: 'loadeddata' means the first frame is visible
      video.addEventListener("loadeddata", hideLoader);
    }
  }
});

async function setRightVideo(idx) {
  idx = ((idx % rightVideos.length) + rightVideos.length) % rightVideos.length;
  const info = rightVideos[idx];
  currentRightIndex = idx;

  const syncTime = videoLeft.currentTime || 0;
  const wasPlaying = !videoLeft.paused && isPlaying;
  const srcPath = `static/videos/${baseView}_comp/${baseView}${info.key}_.mp4`;

  /* --------------------------------------------
     1) Pre-decode using hidden prewarm element
     -------------------------------------------- */
  const pre = document.createElement("video");
  pre.src = srcPath;
  pre.muted = true;
  pre.preload = "auto";
  pre.playsInline = true;
  pre.style.width = "0";
  pre.style.height = "0";
  pre.style.opacity = "0";
  document.body.appendChild(pre);

  // Decode until first real GOP so switch won't freeze
  await new Promise((resolve) => {
    const done = () => {
      pre.removeEventListener("loadeddata", done);
      pre.removeEventListener("canplay", done);
      resolve();
    };
    pre.addEventListener("loadeddata", done, { once: true });
    pre.addEventListener("canplay", done, { once: true });
  });

  // GPU decode warmup via captureStream (forces decode path)
  try {
    const s = pre.captureStream();
    s.getTracks().forEach(t => t.stop());
  } catch(e) {}

  // Cleanup warmup video after decode warm
  setTimeout(() => pre.remove(), 1500);

  /* --------------------------------------------
     2) Switch source WITHOUT load(), WITHOUT pause
     -------------------------------------------- */
  const sourceEl = videoRight.querySelector("source") || document.createElement("source");
  sourceEl.src = srcPath;
  if (!videoRight.querySelector("source")) videoRight.appendChild(sourceEl);

  // DO NOT CALL videoRight.load();
  // DO NOT CALL videoRight.pause();

  /* --------------------------------------------
     3) Fast-seek using once-decoded GOP
     -------------------------------------------- */
  function afterSeek() {
    videoRight.removeEventListener("seeked", afterSeek);
    if (wasPlaying) {
      videoRight.play().catch(()=>{});
      startSync();
    }
  }
  videoRight.addEventListener("seeked", afterSeek);
  videoRight.currentTime = syncTime;

  /* --------------------------------------------
     4) UI text refresh
     -------------------------------------------- */
  const label = baseView === "128" ? info.label128 : info.label32;
  rightLabelEl.textContent = label;
  rightLabelEl.style.color = info.color;
  changeBtn.innerHTML = `Click to compare with <span style="color:${info.change_color}">${info.toggle_label}</span>`;
}