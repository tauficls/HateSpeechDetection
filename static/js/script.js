$(document).ready(function(){
  $('[data-toggle="tooltip"]').tooltip(); 

  $(".navbar a, footer a[href='#home']").on('click', function(event) {
    if (this.hash !== "") {

      event.preventDefault();

      var hash = this.hash;

      $('html, body').animate({
        scrollTop: $(hash).offset().top - 48
      }, 1000, function(){
      });
    }
  });

  $('#read-more').on('click', function () {
    text = $('#read-more').text();
    console.log(text);
    if (text == "read more") {
      $('#read-more').text("read less");
    } else {
      $('#read-more').text("read more");
    }
    $('#more-info').slideToggle(500);
  })
})