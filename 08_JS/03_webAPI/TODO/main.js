const init = () =>{
    addTodo();
};

const todoBox = document.querySelector('#todo_box');
const reverseButton = document.querySelector('#reverse_btn');
const fetchButton = document.querySelector('#fetch_btn');
// TODO: input, Add 버튼에 createTodo와 이벤트
const addTodo = () =>{
    const inputArea = document.querySelector('#add_todo_input');
    const button = document.querySelector('#add_todo_btn');
    inputArea.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            if(inputArea.value === ''){
                alert('공백은 넣을 수 없습니다.');
            }
            else {
                todoBox.appendChild(createTodo(inputArea.value));
                inputArea.value = '';
            }
        }
    });
    button.addEventListener('click', e => {
        if(inputArea.value === ''){
            alert('공백은 넣을 수 없습니다.');
        }
        else {
            todoBox.appendChild(createTodo(inputArea.value));
            inputArea.value = '';
        }
    });
};
reverseButton.addEventListener('click', e =>{
    reverseTodos();
});
fetchButton.addEventListener('click', e => {
    fetchData('https://koreanjson.com/todos');
});
// TODO: 버튼 만들고, 데이터 받아오게 이벤트 리스너
const fetchData = URL =>{
    fetch(URL)
        .then(res=>res.json())
        .then(todos=>{
            for(const todo of todos){
                todoBox.appendChild(createTodo(todo.title, todo.completed));
            }
        })
};

const createTodo = (inputText, completed = false) => {
    // Card
    const todoCard = document.createElement('div');
    todoCard.classList.add('ui', 'segment', 'todo-item');
    if(completed)todoCard.classList.add('secondary');
    // Card > wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'ui checkbox';
    // Card > wrapper > input
    const input = document.createElement('input');
    input.setAttribute('type', 'checkbox');
    input.checked = completed;

    // Card > wrapper > label
    const label = document.createElement('label');
    label.innerHTML = inputText;
    if(completed)label.classList.add('completed-label');

    input.addEventListener('click', e =>{
        if(input.checked){
            todoCard.classList.add('secondary');
            label.classList.add('completed-label');
        }
        else{
            todoCard.classList.remove('secondary');
            label.classList.remove('completed-label');
        }
    });

    const deleteIcon = document.createElement('i');
    deleteIcon.classList.add('close', 'icon', 'delete-icon');
    deleteIcon.addEventListener('click', e =>{
        todoBox.removeChild(todoCard);
    });

    wrapper.appendChild(input);
    wrapper.appendChild(label);
    todoCard.appendChild(wrapper);
    todoCard.appendChild(deleteIcon);
    return todoCard;
};

const reverseTodos = () =>{
    const allTodos = Array.from(document.querySelectorAll('.todo-item'));
    while(todoBox.firstChild){
        todoBox.removeChild(todoBox.firstChild);
    }
    for(const todo of allTodos.reverse()){
        todoBox.appendChild(todo);
    }

};
// todoBox.appendChild(createTodo('hi'));
init();