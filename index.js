

const youtubeIcon = document.querySelector('#youtube i.fa-youtube');

let replacementTimeout;

youtubeIcon.addEventListener('mouseover', () => {
  clearTimeout(replacementTimeout);
  
  const newIcon = document.createElement('i');
  newIcon.className = 'fa-brands fa-youtube fa-shake fa-2xl';
  newIcon.style.color = '#ff0000';
  
  youtubeIcon.replaceWith(newIcon);
  
  replacementTimeout = setTimeout(() => {
    newIcon.replaceWith(youtubeIcon);
  }, 200);
});

youtubeIcon.addEventListener('mouseout', () => {
  clearTimeout(replacementTimeout);
});

const spotofyIcon = document.querySelector('#spotify i.fa-spotify');

let replacementTimeout2;

spotofyIcon.addEventListener('mouseover', () => {
  clearTimeout(replacementTimeout2);
  
  const newIcon = document.createElement('i');
  newIcon.className = 'fa-brands fa-spotify fa-shake fa-2xl';
  newIcon.style.color = '#1DB954';
  
  spotofyIcon.replaceWith(newIcon);
  
  replacementTimeout2 = setTimeout(() => {
    newIcon.replaceWith(spotofyIcon);
  }, 200);
});

spotofyIcon.addEventListener('mouseout', () => {
  clearTimeout(replacementTimeout2);
});

const instagramIcon = document.querySelector('#instagram i.fa-instagram');

let replacementTimeout3;

instagramIcon.addEventListener('mouseover', () => {
  clearTimeout(replacementTimeout3);
  
  const newIcon = document.createElement('i');
  newIcon.className = 'fa-brands fa-instagram fa-shake fa-2xl';
  newIcon.style.color = '#C13584';
  
  instagramIcon.replaceWith(newIcon);
  
  replacementTimeout3 = setTimeout(() => {
    newIcon.replaceWith(instagramIcon);
  }, 200);
});

instagramIcon.addEventListener('mouseout', () => {
  clearTimeout(replacementTimeout3);
});


const twitterIcon = document.querySelector('#twitter i.fa-twitter');

let replacementTimeout4;

twitterIcon.addEventListener('mouseover', () => {
  clearTimeout(replacementTimeout4);
  
  const newIcon = document.createElement('i');
  newIcon.className = 'fa-brands fa-twitter fa-shake fa-2xl';
  newIcon.style.color = '#1DA1F2';
  
  twitterIcon.replaceWith(newIcon);
  
  replacementTimeout4 = setTimeout(() => {
    newIcon.replaceWith(twitterIcon);
  }, 200);
});

twitterIcon.addEventListener('mouseout', () => {
  clearTimeout(replacementTimeout4);
});




// Function to fetch videos from a YouTube channel using the YouTube Data API
async function fetchVideosFromChannel(channelId, apiKey) {
  const apiUrl = `https://www.googleapis.com/youtube/v3/search?part=snippet&channelId=${channelId}&maxResults=50&key=${apiKey}`;
  
  try {
    const response = await fetch(apiUrl);
    const data = await response.json();
    
    return data.items;
  } catch (error) {
    console.error('Error fetching videos:', error);
    return [];
  }
}

// Function to create a video card element for each YouTube video
function createVideoCardElement(video) {
  const card = document.createElement('div');
  card.classList.add('Card1');

  const videoContainer = document.createElement('div');
  videoContainer.classList.add('video-container');

  const videoIframe = document.createElement('iframe');
  videoIframe.src = `https://www.youtube.com/embed/${video.id.videoId}`;
  videoIframe.allowFullscreen = true;
  videoIframe.allow = 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture';
  videoIframe.frameborder = '0';

  videoContainer.appendChild(videoIframe);

  card.appendChild(videoContainer);

  return card;
}

// Your YouTube channel ID and API key
// const apiKey = 'AIzaSyCVFL9-auB4M7IGlU-kEtGqEHVVwDaAIL8';
// const channelId = 'UC2OM05m1q4RSE8eFrQ-oXYQ';

// Fetch videos from the channel and create video card elements
fetchVideosFromChannel(channelId, apiKey)
  .then(videos => {
    const recentPodcastContainer = document.querySelector('.recentpodcast');

    videos.forEach(video => {
      const videoCard = createVideoCardElement(video);
      recentPodcastContainer.appendChild(videoCard);
    });
  })
  .catch(error => {
    console.error('Error:', error);
  });


// Function to pause the scrolling animation
function pauseScrollAnimation() {
  const recentPodcastContainer = document.querySelector('.recentpodcast');
  const cards = recentPodcastContainer.querySelectorAll('.Card1');

  cards.forEach(card => {
    card.style.animationPlayState = 'paused';
  });
}

// Function to resume the scrolling animation
function resumeScrollAnimation() {
  const recentPodcastContainer = document.querySelector('.recentpodcast');
  const cards = recentPodcastContainer.querySelectorAll('.Card1');

  cards.forEach(card => {
    card.style.animationPlayState = 'running';
  });
}

// Add event listeners to pause and resume scrolling animation on mouse enter and leave
const recentPodcastContainer = document.querySelector('.recentpodcast');
recentPodcastContainer.addEventListener('mouseenter', pauseScrollAnimation);
recentPodcastContainer.addEventListener('mouseleave', resumeScrollAnimation);




const microphoneIcon = document.querySelector('.register i.fa-microphone');

microphoneIcon.addEventListener('mouseover', () => {
  microphoneIcon.classList.remove('fa-xl');
  microphoneIcon.classList.add('fa-beat', 'fa-2xl');
});

microphoneIcon.addEventListener('mouseout', () => {
  microphoneIcon.classList.remove('fa-beat', 'fa-2xl');
  microphoneIcon.classList.add('fa-xl');
});

const yearElement = document.getElementById('year');
const currentYear = new Date().getFullYear();
yearElement.textContent = currentYear;


// Get the elements


// Function to handle media query change
function handleMediaQuery(mq) {
  if (mq.matches) {
    // Add classes for small screens
    youtubeIcon.classList.add('fa-2xs');
    spotofyIcon.classList.add('fa-2xs');
    instagramIcon.classList.add('fa-2xs');
    twitterIcon.classList.add('fa-2xs');
  } else {
    // Remove classes for larger screens
    youtubeIcon.classList.remove('fa-2xs');
    spotofyIcon.classList.remove('fa-2xs');
    instagramIcon.classList.remove('fa-2xs');
    twitterIcon.classList.remove('fa-2xs');
  }
}

// Media query for max-width: 600px
const mediaQuery = window.matchMedia('(max-width: 600px)');

// Initial handling of media query
handleMediaQuery(mediaQuery);

// Event listener for media query change
mediaQuery.addEventListener('change', (e) => handleMediaQuery(e.target));

// Event listeners for hover effects
youtubeIcon.addEventListener('mouseenter', () => {
  youtubeIcon.classList.add('fa-xs');
});

youtubeIcon.addEventListener('mouseleave', () => {
  youtubeIcon.classList.remove('fa-xs');
});

spotofyIcon.addEventListener('mouseenter', () => {
  spotofyIcon.classList.add('fa-xs');
});

spotofyIcon.addEventListener('mouseleave', () => {
  spotofyIcon.classList.remove('fa-xs');
});

instagramIcon.addEventListener('mouseenter', () => {
  instagramIcon.classList.add('fa-xs');
});

instagramIcon.addEventListener('mouseleave', () => {
  instagramIcon.classList.remove('fa-xs');
});

twitterIcon.addEventListener('mouseenter', () => {
  twitterIcon.classList.add('fa-xs');
});

twitterIcon.addEventListener('mouseleave', () => {
  twitterIcon.classList.remove('fa-xs');
});

