$(function() {

    $("form").change(function() {
        var formData = new FormData(this);
        $("#result #image").hide();
        $("#result #loading").show();

        $.post({
            url: $(this).attr("action"),
            processData: false,
            data: formData,
            contentType: false
        }).done(function(data) {
            $("#result #loading").hide();
            $("#result #image").attr('src', '/web/' + data.image).show();
            $("#result span").text(data.text);
            $("#result p").show();
        });
    });
});
